use crate::scene;

pub fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_resources: &scene::SceneResources,
    storage_view: &wgpu::TextureView,
    camera_buffer: &wgpu::Buffer,
    accumulation_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&scene_resources.tlas),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scene_resources.material_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: accumulation_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scene_resources.global_vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scene_resources.global_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scene_resources.mesh_info_buffer.as_entire_binding(),
            },
        ],
    })
}

pub fn create_blit_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    storage_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(storage_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
