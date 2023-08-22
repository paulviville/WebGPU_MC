
@group(0) @binding(0)
var <storage, read_write> position : array<vec3f>;

override Xmin : f32 = -1.0;
override Ymin : f32 = -1.0;
override Zmin : f32 = -1.0;
const Min : vec3f = vec3<f32>(-1.0,-1.0,-1.0);
override STEP : f32 = 2.0 / 64.0;
override X : u32 = 32u;
override Y : u32 = 32u;
override Z : u32 = 32u;
override NbVertices = 33u * 33u * 33u;

fn computeVertexId(globalId: u32) -> vec3u {
    var id = globalId;
    let z = id / ((X+1)*(Y+1));
    id %= ((X+1)*(Y+1));
    let y = id / (X+1);
    let x = id % (X+1);
    return vec3u(x, y, z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&position)) {
        return;
    }
    
    let vertexid = computeVertexId(global_id.x);
    position[global_id.x] = vec3f(f32(vertexid.x), f32(vertexid.y), f32(vertexid.z));
    //position[global_id.x] *= STEP;
    //position[global_id.x] += Min;
    // position[global_id.x] = vec3f(f32(global_id.x));
}