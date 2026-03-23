pub mod bind_groups;
pub mod blit_pass;
pub mod build_grid_pass;
pub mod compute_pass;
pub mod debug_photons_pass;
pub mod photon_emit_pass;

use crate::scene;
use crate::utils::wgpu::*;
use crate::wgpu_ctx::WgpuContext;

use blit_pass::BlitPass;
use build_grid_pass::BuildGridPass;
use compute_pass::ComputePass;
use debug_photons_pass::DebugPhotonsPass;
use photon_emit_pass::PhotonEmitPass;

pub const MAX_PHOTONS: u32 = 1024 * 1024;
pub const HASH_SIZE: u32 = 4 * 1024 * 1024;

pub struct Renderer {
    pub photon_emit_pass: PhotonEmitPass,
    pub build_grid_pass: BuildGridPass,
    pub compute_pass: ComputePass,
    pub blit_pass: BlitPass,
    pub debug_photons_pass: DebugPhotonsPass,

    pub sampler: wgpu::Sampler,
    pub storage_texture: wgpu::Texture,
    pub accumulation_buffer: wgpu::Buffer,
    pub frame_count: u32,

    pub photons_buffer: wgpu::Buffer,
    pub photon_count_buffer: wgpu::Buffer,

    pub grid_head_buffer: wgpu::Buffer,
    pub grid_next_buffer: wgpu::Buffer,
    pub clear_head_data: Vec<u32>, // クリア用のデータ

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
        // --- フォトン用バッファの作成 ---
        // Photon構造体は 64 bytes
        let photons_buffer = create_buffer(
            &ctx.device,
            "Photons Buffer",
            (MAX_PHOTONS as u64) * 64,
            wgpu::BufferUsages::STORAGE,
        );
        // カウンター(u32 = 4 bytes)
        let photon_count_buffer = create_buffer(
            &ctx.device,
            "Photon Count Buffer",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // --- パスの初期化 ---
        let photon_emit_pass = PhotonEmitPass::new(
            ctx,
            scene_resources,
            &photons_buffer,
            &photon_count_buffer,
            camera_buffer,
        );

        let grid_head_buffer = create_buffer(
            &ctx.device,
            "Grid Head Buffer",
            (HASH_SIZE as u64) * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let grid_next_buffer = create_buffer(
            &ctx.device,
            "Grid Next Buffer",
            (HASH_SIZE as u64) * 4,
            wgpu::BufferUsages::STORAGE,
        );

        let build_grid_pass = BuildGridPass::new(
            ctx,
            &photons_buffer,
            &photon_count_buffer,
            &grid_head_buffer,
            &grid_next_buffer,
        );

        // --- レイトレース用バッファの作成 ---
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
            &photons_buffer,
            &grid_head_buffer,
            &grid_next_buffer,
        );

        let blit_pass = BlitPass::new(ctx, &storage_view, &sampler);

        let debug_photons_pass = DebugPhotonsPass::new(ctx, &photons_buffer, camera_buffer);

        Self {
            photon_emit_pass,
            build_grid_pass,
            compute_pass,
            blit_pass,
            debug_photons_pass,
            sampler,
            storage_texture,
            accumulation_buffer,
            frame_count: 0,
            photons_buffer,
            photon_count_buffer,
            grid_head_buffer,
            grid_next_buffer,
            clear_head_data: vec![u32::MAX; HASH_SIZE as usize],
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

        ctx.queue
            .write_buffer(&self.photon_count_buffer, 0, bytemuck::cast_slice(&[0u32]));
        ctx.queue.write_buffer(
            &self.grid_head_buffer,
            0,
            bytemuck::cast_slice(&self.clear_head_data),
        );

        // 1. Photon Emit Pass
        self.photon_emit_pass.record(&mut encoder, MAX_PHOTONS);

        // 2. Build Grid Pass
        self.build_grid_pass.record(&mut encoder, MAX_PHOTONS);

        // 3. Compute Pass
        self.compute_pass
            .record(&mut encoder, self.target_width, self.target_height);

        // 4. Render Pass (Blit)
        self.blit_pass.record(&mut encoder, view);

        // 5. Debug Pass (Photons Overlay)
        // Blitされた絵の上に、フォトンを重ねて描画する
        // self.debug_photons_pass
        //     .record(&mut encoder, view, MAX_PHOTONS);

        ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_count += 1;

        Ok(())
    }
}
