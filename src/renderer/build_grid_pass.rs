use crate::renderer::bind_groups::*;
use crate::wgpu_ctx::WgpuContext;

pub struct BuildGridPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
}

impl BuildGridPass {
    pub fn new(
        ctx: &WgpuContext,
        photons_buffer: &wgpu::Buffer,
        photon_count_buffer: &wgpu::Buffer,
        grid_head_buffer: &wgpu::Buffer,
        grid_next_buffer: &wgpu::Buffer,
        camera_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/build_grid.wgsl"));

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Build Grid Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            // photons
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            // photon_count
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            // grid_head
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            // grid_next
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            // camera
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                    ],
                });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Build Grid Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bind_group_layout],
                            immediate_size: 0,
                        },
                    )),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let bind_group = create_build_grid_bind_group(
            &ctx.device,
            &bind_group_layout,
            photons_buffer,
            photon_count_buffer,
            grid_head_buffer,
            grid_next_buffer,
            camera_buffer,
        );

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, max_photons: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Build Grid Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups((max_photons + 63) / 64, 1, 1);
    }
}
