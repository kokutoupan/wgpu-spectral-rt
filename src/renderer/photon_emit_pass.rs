use crate::renderer::bind_groups::*;
use crate::scene;
use crate::wgpu_ctx::WgpuContext;

pub struct PhotonEmitPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
}

impl PhotonEmitPass {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        photons_buffer: &wgpu::Buffer,
        photon_count_buffer: &wgpu::Buffer,
        camera_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/photon_emit.wgsl"));

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Photon Emit Bind Group Layout"),
                    entries: &[
                        // 0: TLAS
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::AccelerationStructure {
                                vertex_return: false,
                            },
                            count: None,
                        },
                        // 1: Lights Buffer (Read Only)
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
                        // 2: Photons Buffer (Read/Write)
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
                        // 3: Photon Count (Atomic)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 4: Camera (Uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 5: Materials
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        // 6: Vertices
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        // 7: Indices
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        // 8: MeshInfos
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                    ],
                });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Photon Emit Pipeline"),
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

        let bind_group = create_photon_emit_bind_group(
            &ctx.device,
            &bind_group_layout,
            scene_resources,
            photons_buffer,
            photon_count_buffer,
            camera_buffer,
        );

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, max_photons: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Photon Emit Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        // Workgroup sizeを64としているので、必要なグループ数を計算
        cpass.dispatch_workgroups((max_photons + 63) / 64, 1, 1);
    }
}
