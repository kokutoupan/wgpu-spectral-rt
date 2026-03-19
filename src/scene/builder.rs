use crate::scene::geometry::Vertex;
use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub extra: [f32; 4], // x: type, y: fuzz, z: ior, w: cauchy_a
}

// 光源情報
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightInfo {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub normal: [f32; 4],
    pub params: [f32; 4], // [0]:type, [1]:temp, [2]:intensity, [3]:pad
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInfo {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub pad: [u32; 2],
}

pub struct SceneResources {
    pub tlas: wgpu::Tlas,
    pub global_vertex_buffer: wgpu::Buffer,
    pub global_index_buffer: wgpu::Buffer,
    pub mesh_info_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub light_buffer: wgpu::Buffer,
    // 個別のBLASはTLAS構築後に破棄しても動きますが、念のため保持
    _blases: Vec<wgpu::Blas>,
}

pub struct InstanceData {
    pub mesh_id: u32,
    pub mat_id: u32,
    pub transform: Mat4,
}

pub struct SceneBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    mesh_infos: Vec<MeshInfo>,
    materials: Vec<MaterialUniform>,
    instances: Vec<InstanceData>,
    meshes: Vec<(Vec<Vertex>, Vec<u32>)>, // CPU側のメッシュデータ
    lights: Vec<LightInfo>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            mesh_infos: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            meshes: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn add_material(&mut self, material: MaterialUniform) -> u32 {
        let id = self.materials.len() as u32;
        self.materials.push(material);
        id
    }

    pub fn add_mesh(&mut self, vertices: &[Vertex], indices: &[u32]) -> u32 {
        let mesh_id = self.meshes.len() as u32;

        // Global Buffer 用のオフセットを記録
        self.mesh_infos.push(MeshInfo {
            vertex_offset: self.vertices.len() as u32,
            index_offset: self.indices.len() as u32,
            pad: [0; 2],
        });

        self.vertices.extend_from_slice(vertices);
        self.indices.extend_from_slice(indices);
        self.meshes.push((vertices.to_vec(), indices.to_vec()));

        mesh_id
    }

    pub fn add_instance(&mut self, mesh_id: u32, mat_id: u32, transform: Mat4) {
        self.instances.push(InstanceData {
            mesh_id,
            mat_id,
            transform,
        });
    }

    pub fn add_light_instance(
        &mut self,
        mesh_id: u32,
        mat_id: u32,
        transform: Mat4,
        params: [f32; 4],
    ) {
        self.add_instance(mesh_id, mat_id, transform);

        let (verts, inds) = &self.meshes[mesh_id as usize];
        // ポリゴン(三角形)ごとにLightInfoを生成
        for i in (0..inds.len()).step_by(3) {
            let v0 = verts[inds[i] as usize].pos;
            let v1 = verts[inds[i + 1] as usize].pos;
            let v2 = verts[inds[i + 2] as usize].pos;

            // ローカル座標からワールド座標へ変換
            let p0 = transform * Vec4::new(v0[0], v0[1], v0[2], 1.0);
            let p1 = transform * Vec4::new(v1[0], v1[1], v1[2], 1.0);
            let p2 = transform * Vec4::new(v2[0], v2[1], v2[2], 1.0);

            // 法線を計算
            let edge1 = p1.truncate() - p0.truncate();
            let edge2 = p2.truncate() - p0.truncate();
            let normal = edge1.cross(edge2).normalize();

            self.lights.push(LightInfo {
                v0: [p0.x, p0.y, p0.z, 1.0],
                v1: [p1.x, p1.y, p1.z, 1.0],
                v2: [p2.x, p2.y, p2.z, 1.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
                params,
            });
        }
    }

    pub fn build(self, device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut blases = Vec::new();

        // 1. 各メッシュのBLASを作成
        for (i, (verts, inds)) in self.meshes.iter().enumerate() {
            let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("BLAS Vertex Buf {}", i)),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("BLAS Index Buf {}", i)),
                contents: bytemuck::cast_slice(inds),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });

            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: verts.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(inds.len() as u32),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };

            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some(&format!("BLAS {}", i)),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size_desc.clone()],
                },
            );

            // Rustのライフタイム制約を回避するため、エンコーダへの登録は後で一括で行う
            blases.push((blas, v_buf, i_buf, size_desc));
        }

        for (blas, v_buf, i_buf, size_desc) in &blases {
            encoder.build_acceleration_structures(
                Some(&wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: size_desc,
                            vertex_buffer: v_buf,
                            first_vertex: 0,
                            vertex_stride: std::mem::size_of::<Vertex>() as u64,
                            index_buffer: Some(i_buf),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }),
                None,
            );
        }

        // 2. Global Buffers の作成
        let global_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let global_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mesh_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Info Buffer"),
            contents: bytemuck::cast_slice(&self.mesh_infos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Buffer"),
            contents: bytemuck::cast_slice(&self.materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&if self.lights.is_empty() {
                // 空バッファエラー回避用のダミー
                vec![LightInfo {
                    v0: [0.; 4],
                    v1: [0.; 4],
                    v2: [0.; 4],
                    normal: [0.; 4],
                    params: [0.; 4],
                }]
            } else {
                self.lights.clone()
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 3. TLAS の作成
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Scene TLAS"),
            max_instances: self.instances.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        let mut tlas_instances = Vec::new();
        for inst in &self.instances {
            let affine = inst.transform.transpose().to_cols_array();
            let custom_data = (inst.mesh_id << 16) | inst.mat_id; // Mesh ID と Material ID をパック

            tlas_instances.push(Some(wgpu::TlasInstance::new(
                &blases[inst.mesh_id as usize].0,
                affine[..12].try_into().unwrap(),
                custom_data,
                0xff,
            )));
        }

        // TLASインスタンスをセットしてビルド
        for (i, inst) in tlas_instances.into_iter().enumerate() {
            tlas[i] = inst;
        }
        encoder.build_acceleration_structures(None, Some(&tlas));
        queue.submit(std::iter::once(encoder.finish()));

        SceneResources {
            tlas,
            global_vertex_buffer,
            global_index_buffer,
            mesh_info_buffer,
            material_buffer,
            light_buffer,
            _blases: blases.into_iter().map(|b| b.0).collect(),
        }
    }
}
