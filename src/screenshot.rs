use image::ImageEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

pub struct ScreenshotTask {
    pub width: u32,
    pub height: u32,
    pub padded_bytes_per_row: u32,
    pub data: Vec<u8>,
}

pub struct ScreenshotSaver {
    // Buffers for reuse
    image_data: Vec<u8>,
}

impl ScreenshotSaver {
    pub fn new() -> Self {
        Self {
            image_data: Vec::new(),
        }
    }

    pub fn process_and_save(&mut self, task: ScreenshotTask) {
        let ScreenshotTask {
            width,
            height,
            padded_bytes_per_row,
            data: raw_data,
        } = task;

        let saving_start = chrono::Local::now();
        let unpadded_bytes_per_row = (width * 4) as usize;
        let padded_bytes_per_row = padded_bytes_per_row as usize;
        let pixel_count = (width * height) as usize;

        println!("Process Mode: Fast Blur (High Speed)");

        // 1. パディング除去 & BGRA->RGBA変換 & u8生成 (一撃で行う)
        // self.image_dataを再利用 (rgba_data は削除して一本化)
        if self.image_data.len() != pixel_count * 4 {
            self.image_data.resize(pixel_count * 4, 0);
        }

        self.image_data
            .par_chunks_mut(unpadded_bytes_per_row)
            .zip(raw_data.par_chunks(padded_bytes_per_row))
            .for_each(|(dest_row, src_row)| {
                for (dest_pixel, src_pixel) in
                    dest_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4))
                {
                    dest_pixel[0] = src_pixel[2]; // R (Src:B)
                    dest_pixel[1] = src_pixel[1]; // G (Src:G)
                    dest_pixel[2] = src_pixel[0]; // B (Src:R)
                    dest_pixel[3] = 255; // A
                }
            });

        let now = chrono::Local::now();
        let filename = format!("output/screenshot_{}.png", now.format("%Y-%m-%d_%H-%M-%S"));
        let _ = std::fs::create_dir_all("output");

        let file = File::create(&filename).unwrap();
        let ref mut w = BufWriter::new(file);

        let encoder = PngEncoder::new_with_quality(
            w,
            CompressionType::Fast, // 爆速設定
            FilterType::NoFilter,
        );

        encoder
            .write_image(
                &self.image_data, // 直接スライスを渡す
                width,
                height,
                image::ColorType::Rgba8.into(),
            )
            .unwrap();

        println!(
            "Saved screenshot: {} ({}ms)",
            filename,
            chrono::Local::now().timestamp_millis() - saving_start.timestamp_millis()
        );
    }
}
