use std::sync::Arc;
use winit::window::Window;

use crate::renderer::Renderer;
use crate::scene;
use crate::scene::CameraController;
use crate::screenshot::ScreenshotTask;
use crate::utils::wgpu::*;
use crate::wgpu_ctx::WgpuContext;

pub struct Engine {
    ctx: WgpuContext,
    window: Arc<Window>,

    // レンダリングコア
    renderer: Renderer,
    scene_resources: scene::SceneResources,

    // カメラ
    camera_buffer: wgpu::Buffer,
    camera_controller: CameraController,

    // 状態管理
    is_paused: bool,
    screenshot_requested: bool,
    screenshot_buffer: wgpu::Buffer,
    screenshot_padded_bytes_per_row: u32,
    screenshot_sender: std::sync::mpsc::Sender<ScreenshotTask>,

    // デルタタイム計算用
    last_frame_time: std::time::Instant,
    fps_timer: f32,
    fps_frame_count: u32,
}

impl Engine {
    pub async fn new(window: Arc<Window>, target_width: u32, target_height: u32) -> Self {
        // スクリーンショット用スレッドの起動
        let (screenshot_sender, screenshot_receiver) = std::sync::mpsc::channel::<ScreenshotTask>();
        std::thread::spawn(move || {
            let mut saver = crate::screenshot::ScreenshotSaver::new();
            while let Ok(task) = screenshot_receiver.recv() {
                saver.process_and_save(task);
            }
        });

        let ctx = WgpuContext::new(window.clone()).await;

        // シーンとカメラの初期化
        let scene_resources = scene::create_cornell_box(&ctx.device, &ctx.queue);
        // let scene_resources = scene::create_custom_scene(&ctx.device, &ctx.queue);
        let camera_controller = CameraController::new();

        let camera_uniform =
            camera_controller.build_uniform(target_width as f32 / target_height as f32, 0);

        let camera_buffer = create_buffer_init(
            &ctx.device,
            "Camera Buffer",
            &[camera_uniform],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let renderer = Renderer::new(
            &ctx,
            &scene_resources,
            &camera_buffer,
            target_width,
            target_height,
        );

        // スクリーンショットバッファの準備 (レンダリングターゲットの解像度を使用)
        let screenshot_padded_bytes_per_row = get_padded_bytes_per_row(target_width);
        let screenshot_buffer = create_buffer(
            &ctx.device,
            "Screenshot Buffer",
            (screenshot_padded_bytes_per_row * target_height) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        Self {
            ctx,
            window,
            renderer,
            scene_resources,
            camera_buffer,
            camera_controller,
            is_paused: false,
            screenshot_requested: false,
            screenshot_buffer,
            screenshot_padded_bytes_per_row,
            screenshot_sender,
            last_frame_time: std::time::Instant::now(),
            fps_timer: 0.0,
            fps_frame_count: 0,
        }
    }

    // app.rs からウィンドウにアクセスするためのゲッター
    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.ctx.resize(new_size);
            self.renderer
                .resize(&self.ctx, &self.scene_resources, &self.camera_buffer);

            // スクリーンショットバッファは固定解像度なのでリサイズ不要
        }
    }

    pub fn input(&mut self, event: &winit::event::WindowEvent) {
        if let winit::event::WindowEvent::KeyboardInput { event: ev, .. } = event {
            if ev.state == winit::event::ElementState::Pressed && !ev.repeat {
                match ev.logical_key {
                    winit::keyboard::Key::Character(ref s) if s == "j" => {
                        self.is_paused = !self.is_paused;
                    }
                    winit::keyboard::Key::Character(ref s) if s == "k" => {
                        self.screenshot_requested = true;
                    }
                    _ => (),
                }
            }
        }
        self.camera_controller.process_events(event);
    }

    pub fn update(&mut self) {
        if self.is_paused {
            return;
        }

        let now = std::time::Instant::now();
        let dt = now - self.last_frame_time;
        self.last_frame_time = now;

        let dt_secs = dt.as_secs_f32();
        self.fps_timer += dt_secs;
        self.fps_frame_count += 1;

        if self.fps_timer >= 0.5 {
            let fps = self.fps_frame_count as f32 / self.fps_timer;
            self.window.set_title(&format!(
                "wgpu-spectral-rt | FPS: {:.1} | SPP: {} | Res: {}x{}",
                fps, self.renderer.frame_count, self.ctx.config.width, self.ctx.config.height
            ));
            self.fps_timer = 0.0;
            self.fps_frame_count = 0;
        }

        if self.camera_controller.update_camera(dt) {
            self.renderer.frame_count = 0;
        }

        let camera_uniform = self.camera_controller.build_uniform(
            self.renderer.target_width as f32 / self.renderer.target_height as f32,
            self.renderer.frame_count,
        );
        self.ctx.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.is_paused {
            return Ok(());
        }

        let output = self.ctx.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render(&self.ctx, &view)?;

        if self.screenshot_requested {
            self.save_screenshot();
            self.screenshot_requested = false;
        }

        output.present();

        Ok(())
    }

    fn save_screenshot(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.renderer.storage_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.screenshot_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.screenshot_padded_bytes_per_row),
                    rows_per_image: Some(self.renderer.target_height),
                },
            },
            wgpu::Extent3d {
                width: self.renderer.target_width,
                height: self.renderer.target_height,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // GPUの処理待ち & 読み取り
        let buffer_slice = self.screenshot_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Ok(Ok(_)) = rx.recv() {
            let data = buffer_slice.get_mapped_range().to_vec();
            self.screenshot_buffer.unmap();

            let task = ScreenshotTask {
                data,
                width: self.renderer.target_width,
                height: self.renderer.target_height,
                padded_bytes_per_row: self.screenshot_padded_bytes_per_row,
                spp: self.renderer.frame_count,
            };
            self.screenshot_sender.send(task).unwrap();
        }
    }
}
