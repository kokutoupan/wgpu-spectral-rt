use crate::utils::wgpu::*;
use crate::wgpu_ctx::WgpuContext;
use crate::renderer::bind_groups::*;

pub struct BlitPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
}

impl BlitPass {
    pub fn new(
        ctx: &WgpuContext,
        storage_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/blit.wgsl"));

        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Blit Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bind_group_layout],
                            immediate_size: 0,
                        },
                    )),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL, // Don't clear outside the draw region
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

        // Initialize blit uniform empty (will be overwritten on first frame)
        let uniform_buffer = create_buffer_init(
            &ctx.device,
            "Blit Uniform Buffer",
            &[0.0f32, 0.0, 0.0, 0.0],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = create_blit_bind_group(
            &ctx.device,
            &bind_group_layout,
            storage_view,
            sampler,
            &uniform_buffer,
        );

        Self {
            pipeline,
            bind_group,
            bind_group_layout,
            uniform_buffer,
        }
    }

    pub fn resize(
        &mut self,
        ctx: &WgpuContext,
        target_width: u32,
        target_height: u32,
        storage_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let target_aspect = target_width as f32 / target_height as f32;
        let window_aspect = ctx.config.width as f32 / ctx.config.height as f32;

        let mut scale_x = 1.0;
        let mut scale_y = 1.0;

        if window_aspect > target_aspect {
            // Window is wider than target. Scale X to fit height.
            scale_x = target_aspect / window_aspect;
        } else {
            // Window is taller than target. Scale Y to fit width.
            scale_y = window_aspect / target_aspect;
        }

        let offset_x = 0.0;
        let offset_y = 0.0;

        let scale_offset = [scale_x, scale_y, offset_x, offset_y];

        ctx.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&scale_offset),
        );

        self.bind_group = create_blit_bind_group(
            &ctx.device,
            &self.bind_group_layout,
            storage_view,
            sampler,
            &self.uniform_buffer,
        );
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
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
        rpass.draw(0..6, 0..1);
    }
}
