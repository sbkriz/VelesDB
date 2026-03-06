//! GPU-accelerated k-means assignment for PQ training.
//!
//! Provides optional GPU acceleration for the k-means assignment step during
//! PQ codebook training. Falls back silently to CPU when GPU is unavailable
//! or the dataset is too small to benefit from GPU dispatch.

use wgpu::util::DeviceExt;

/// WGSL compute shader for PQ k-means assignment.
///
/// For each vector, finds the nearest centroid by L2 distance.
const PQ_KMEANS_ASSIGN_SHADER: &str = r"
struct Params {
    num_vectors: u32,
    num_centroids: u32,
    subspace_dim: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> vectors: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn kmeans_assign(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_vectors) { return; }

    let sd = params.subspace_dim;
    let k = params.num_centroids;
    let vec_offset = idx * sd;

    var best_dist: f32 = 3.4028235e+38;
    var best_idx: u32 = 0u;

    for (var c: u32 = 0u; c < k; c = c + 1u) {
        let cent_offset = c * sd;
        var dist: f32 = 0.0;
        for (var d: u32 = 0u; d < sd; d = d + 1u) {
            let diff = vectors[vec_offset + d] - centroids[cent_offset + d];
            dist = dist + diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = c;
        }
    }
    assignments[idx] = best_idx;
}
";

/// Check if GPU dispatch is worthwhile based on FLOP threshold.
///
/// GPU dispatch overhead (device init, buffer copies, kernel launch) is only
/// amortized when the computation exceeds ~10M FLOPs.
#[must_use]
pub fn should_use_gpu(n: usize, k: usize, subspace_dim: usize) -> bool {
    n * k * subspace_dim > 10_000_000
}

/// Compute k-means assignments on GPU for one subspace.
///
/// For each vector, finds the nearest centroid index using L2 distance.
/// Returns assignment indices (one per vector), or `None` if GPU dispatch fails.
///
/// Falls back to `None` if:
/// - wgpu device initialization fails
/// - Buffer creation fails
/// - Compute dispatch fails
/// - Result readback fails
#[allow(clippy::too_many_lines)]
pub fn gpu_kmeans_assign(
    sub_vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
    subspace_dim: usize,
) -> Option<Vec<usize>> {
    if sub_vectors.is_empty() || centroids.is_empty() || subspace_dim == 0 {
        return None;
    }

    let n = sub_vectors.len();
    let k = centroids.len();

    // Initialize GPU device
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("VelesDB PQ K-means"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    // Flatten vectors and centroids into contiguous buffers
    let mut flat_vectors = Vec::with_capacity(n * subspace_dim);
    for v in sub_vectors {
        flat_vectors.extend_from_slice(v);
    }

    let mut flat_centroids = Vec::with_capacity(k * subspace_dim);
    for c in centroids {
        flat_centroids.extend_from_slice(c);
    }

    // Create shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PQ K-means Assignment Shader"),
        source: wgpu::ShaderSource::Wgsl(PQ_KMEANS_ASSIGN_SHADER.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("PQ K-means Bind Group Layout"),
        entries: &[
            // Vectors buffer (binding 0)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Centroids buffer (binding 1)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Assignments buffer (binding 2)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Params uniform (binding 3)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PQ K-means Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("PQ K-means Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("kmeans_assign"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Create buffers
    let vectors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vectors Buffer"),
        contents: bytemuck::cast_slice(&flat_vectors),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let centroids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Centroids Buffer"),
        contents: bytemuck::cast_slice(&flat_centroids),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let assignments_size = (n * std::mem::size_of::<u32>()) as u64;
    let assignments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Assignments Buffer"),
        size: assignments_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: assignments_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Params: [num_vectors, num_centroids, subspace_dim, padding]
    // SAFETY: n, k, and subspace_dim are validated to be non-zero above.
    // In practice these are bounded by training set size (thousands) and
    // centroid count (<=65535), well within u32 range.
    #[allow(clippy::cast_possible_truncation)]
    let params = [n as u32, k as u32, subspace_dim as u32, 0_u32];
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PQ K-means Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vectors_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: centroids_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: assignments_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("PQ K-means Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PQ K-means Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // SAFETY: n is bounded by training set size. div_ceil(256) reduces
        // the value further. Even 4B vectors / 256 = 16M workgroups, fitting in u32.
        #[allow(clippy::cast_possible_truncation)]
        let workgroups = n.div_ceil(256) as u32;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&assignments_buffer, 0, &staging_buffer, 0, assignments_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    rx.recv().ok().and_then(Result::ok)?;

    let data = buffer_slice.get_mapped_range();
    let assignments_u32: &[u32] = bytemuck::cast_slice(&data);
    let assignments: Vec<usize> = assignments_u32.iter().map(|&a| a as usize).collect();
    drop(data);
    staging_buffer.unmap();

    Some(assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_gpu_threshold() {
        // Below threshold: 100 * 16 * 8 = 12800 < 10M
        assert!(!should_use_gpu(100, 16, 8));

        // Above threshold: 10000 * 256 * 8 = 20_480_000 > 10M
        assert!(should_use_gpu(10000, 256, 8));

        // Exactly at threshold: should not trigger (strictly greater)
        assert!(!should_use_gpu(10_000_000 / (256 * 8), 256, 8));
    }

    #[test]
    fn test_gpu_kmeans_assign_matches_cpu() {
        // Small dataset: 10 vectors, 3 centroids, dim=4
        let sub_vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.5],
        ];
        let centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // CPU assignments (nearest centroid by L2)
        let cpu_assignments: Vec<usize> = sub_vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| {
                        let dist: f32 =
                            v.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                        (idx, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        // GPU assignments (may return None if no GPU available)
        if let Some(gpu_assignments) = gpu_kmeans_assign(&sub_vectors, &centroids, 4) {
            assert_eq!(
                gpu_assignments.len(),
                sub_vectors.len(),
                "GPU must return one assignment per vector"
            );
            assert_eq!(
                gpu_assignments, cpu_assignments,
                "GPU assignments must match CPU"
            );
        }
        // If GPU is not available, test passes silently (fallback behavior)
    }

    #[test]
    fn test_gpu_kmeans_assign_empty_input() {
        assert!(gpu_kmeans_assign(&[], &[vec![1.0]], 1).is_none());
        assert!(gpu_kmeans_assign(&[vec![1.0]], &[], 1).is_none());
        assert!(gpu_kmeans_assign(&[vec![1.0]], &[vec![1.0]], 0).is_none());
    }
}
