struct Vertex {
    position: vec3f,
    value: f32,
}

struct VertexInput {
    @builtin(instance_index) instance : u32,
    // @location(0) position : vec3f,
    @builtin(vertex_index) vertexIndex : u32,
}

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f,
}

struct Camera {
    @size(16) position: vec3f,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

struct Uniforms {
    camera: Camera,
    //model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var <storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var <storage, read> edgesId : array<u32>;

const INVALID = 0u - 1u;
const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 16.0;
override X : u32 = 16u;
override Y : u32 = 16u;
override Z : u32 = 16u;
override NbVertices = (X+1) * (Y+1) * (Z+1);

fn vertexId(coord: vec3u) -> u32{
    return coord.x + (X+1) * coord.y + (X+1)*(Y+1) * coord.z;
}

override nbX = (X)*(Y+1)*(Z+1);
override nbY = (X+1)*(Y)*(Z+1);
override nbZ = (X+1)*(Y+1)*(Z);

fn edgeCell(e: u32) -> vec3u {
    let ABC = vec3u(
        u32(e >= nbX) * nbX + u32(e >= (nbX + nbY)) * nbY,
        (X + u32(e > nbX)) * (Y + u32((e < nbX) || (e > nbX + nbY))),
        X + u32(e > nbX)
    );

    var xyz : vec3u;
    var e_ = e - ABC.x;
    xyz.z = e_ / ABC.y;
    e_ = e_ % ABC.y;
    xyz.y = e_ / ABC.z;
    xyz.x = e_ % ABC.z;

    return xyz;
}


@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;
    if(edgesId[input.instance] == INVALID) {
        return output;
    }

    let cell = edgeCell(input.instance);
    let axis = vec3u(
        u32(input.instance < nbX),
        u32((input.instance < nbY + nbX) && input.instance >= nbX),
        u32(input.instance >= (nbX + nbY))
    );

    let vert = vertices[vertexId(cell + input.vertexIndex * axis)];
    output.position = vec4f(vert.position, 1.0);
    output.position = uniforms.camera.projection * uniforms.camera.view * output.position;
    output.color = vec4f(0.0, 1.0, 0.0, 1.0);

    return output;
}

@fragment
fn fragment(@location(0) color: vec4f) -> @location(0) vec4f {
    if(color.g == 0.0) {
        discard;
    }
    return color;
}