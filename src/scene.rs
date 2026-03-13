use crate::geometry;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub use crate::geometry::Vertex;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub color: [f32; 4],
    pub emission: [f32; 4],
    pub extra: [f32; 4], // x: type, y: fuzz, z: ior, w: padding
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
    // 個別のBLASはTLAS構築後に破棄しても動きますが、念のため保持
    _blases: Vec<wgpu::Blas>,
}

struct InstanceData {
    mesh_id: u32,
    mat_id: u32,
    transform: Mat4,
}

pub struct SceneBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    mesh_infos: Vec<MeshInfo>,
    materials: Vec<MaterialUniform>,
    instances: Vec<InstanceData>,
    meshes: Vec<(Vec<Vertex>, Vec<u32>)>, // CPU側のメッシュデータ
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
            _blases: blases.into_iter().map(|b| b.0).collect(),
        }
    }
}

// 構築の呼び出し側 (コーネルボックス)
pub fn create_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneResources {
    let mut builder = SceneBuilder::new();

    // マテリアルの登録 (戻り値がIDになる)
    let mat_light = builder.add_material(MaterialUniform {
        color: [0., 0., 0., 1.],
        emission: [10., 10., 10., 1.],
        extra: [3., 0., 0., 0.],
    });
    let mat_red = builder.add_material(MaterialUniform {
        color: [0.65, 0.05, 0.05, 1.],
        emission: [0., 0., 0., 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_green = builder.add_material(MaterialUniform {
        color: [0.12, 0.45, 0.15, 1.],
        emission: [0., 0., 0., 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_white = builder.add_material(MaterialUniform {
        color: [0.73, 0.73, 0.73, 1.],
        emission: [0., 0., 0., 1.],
        extra: [0., 0., 0., 0.],
    });
    let mat_glass = builder.add_material(MaterialUniform {
        color: [1., 1., 1., 1.],
        emission: [0., 0., 0., 0.],
        extra: [2., 0., 1.5, 0.],
    });
    let mat_metal = builder.add_material(MaterialUniform {
        color: [0.8, 0.8, 0.8, 1.],
        emission: [0., 0., 0., 1.],
        extra: [1., 0.2, 0., 0.],
    });

    let (plane_v, plane_i) = geometry::create_plane();
    let (cube_v, cube_i) = geometry::create_cube();
    let (sphere_v, sphere_i) = geometry::create_sphere(3);

    let mesh_plane = builder.add_mesh(&plane_v, &plane_i);
    let mesh_cube = builder.add_mesh(&cube_v, &cube_i);
    let mesh_sphere = builder.add_mesh(&sphere_v, &sphere_i);

    // インスタンスの配置
    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., -1., 0.)) * Mat4::from_scale(Vec3::splat(2.)),
    ); // Floor
    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., 1., 0.))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(2.)),
    ); // Ceil
    builder.add_instance(
        mesh_plane,
        mat_white,
        Mat4::from_translation(Vec3::new(0., 0., -1.))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    ); // Back
    builder.add_instance(
        mesh_plane,
        mat_red,
        Mat4::from_translation(Vec3::new(-1., 0., 0.))
            * Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    ); // Left
    builder.add_instance(
        mesh_plane,
        mat_green,
        Mat4::from_translation(Vec3::new(1., 0., 0.))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(2.)),
    ); // Right
    builder.add_instance(
        mesh_plane,
        mat_light,
        Mat4::from_translation(Vec3::new(0., 0.99, 0.))
            * Mat4::from_rotation_x(std::f32::consts::PI)
            * Mat4::from_scale(Vec3::splat(0.5)),
    ); // Light

    builder.add_instance(
        mesh_cube,
        mat_metal,
        Mat4::from_translation(Vec3::new(-0.35, -0.4 + 0.002, -0.3))
            * Mat4::from_rotation_y(0.4)
            * Mat4::from_scale(Vec3::new(0.6, 1.2, 0.6)),
    );
    builder.add_instance(
        mesh_sphere,
        mat_glass,
        Mat4::from_translation(Vec3::new(0.4, -0.65, 0.3)) * Mat4::from_scale(Vec3::splat(0.75)),
    );

    // 一気にビルドして返す
    builder.build(device, queue)
}
