pub mod bind_groups;
pub mod blit_pass;
pub mod compute_pass;

use crate::scene;
use crate::utils::wgpu::*;
use crate::wgpu_ctx::WgpuContext;

use blit_pass::BlitPass;
use compute_pass::ComputePass;

pub struct Renderer {
    pub compute_pass: ComputePass,
    pub blit_pass: BlitPass,

    pub sampler: wgpu::Sampler,
    pub storage_texture: wgpu::Texture,
    pub accumulation_buffer: wgpu::Buffer,
    pub frame_count: u32,

    pub target_width: u32,
    pub target_height: u32,
}

impl Renderer {
    pub fn new(
        ctx: &WgpuContext,
        scene_resources: &scene::SceneResources,
        camera_buffer: &wgpu::Buffer,
        target_width: u32,
        target_height: u32,
    ) -> Self {
        let accumulation_buffer = create_buffer(
            &ctx.device,
            "Accumulation Buffer",
            (target_width * target_height * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let storage_texture = create_storage_texture(&ctx.device, target_width, target_height);
        let storage_view = storage_texture.create_view(&Default::default());

        let sampler = ctx
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());

        let compute_pass = ComputePass::new(
            ctx,
            scene_resources,
            &storage_view,
            camera_buffer,
            &accumulation_buffer,
        );

        let blit_pass = BlitPass::new(ctx, &storage_view, &sampler);

        Self {
            compute_pass,
            blit_pass,
            sampler,
            storage_texture,
            accumulation_buffer,
            frame_count: 0,
            target_width,
            target_height,
        }
    }

    pub fn resize(
        &mut self,
        ctx: &WgpuContext,
        _scene_resources: &scene::SceneResources,
        _camera_buffer: &wgpu::Buffer,
    ) {
        let storage_view = self.storage_texture.create_view(&Default::default());

        self.blit_pass.resize(
            ctx,
            self.target_width,
            self.target_height,
            &storage_view,
            &self.sampler,
        );

        // Don't reset frame count on window resize, since the underlying render hasn't changed viewpoint
    }

    pub fn render(
        &mut self,
        ctx: &WgpuContext,
        view: &wgpu::TextureView,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 1. Compute Pass
        self.compute_pass
            .record(&mut encoder, self.target_width, self.target_height);

        // 2. Render Pass (Blit)
        self.blit_pass.record(&mut encoder, view);

        ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;

        Ok(())
    }
}
