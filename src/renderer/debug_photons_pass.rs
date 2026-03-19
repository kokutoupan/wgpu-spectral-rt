use crate::wgpu_ctx::WgpuContext;

pub struct DebugPhotonsPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
}

impl DebugPhotonsPass {
    pub fn new(
        ctx: &WgpuContext,
        photons_buffer: &wgpu::Buffer,
        camera_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/debug_photons.wgsl"));

        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Debug Photons Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // Camera
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        // Photons
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Debug Photons BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: photons_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Debug Photons Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&layout],
                            immediate_size: 0,
                        },
                    )),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.config.format,
                            // 加算合成 (Additive Blending) の設定
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::PointList, // ★ここがポイント！ポリゴンではなく点を打つ
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        max_photons: u32,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Debug Photons Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                // 背景をクリアせず、BlitPassの絵の上に上書き(LoadOp::Load)する
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..max_photons, 0..1);
    }
}
