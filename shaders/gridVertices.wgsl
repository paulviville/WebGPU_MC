struct Vertex {
    position: vec3f,
    value: f32,
}

@group(0) @binding(0)
var <storage, read_write> vertices : array<Vertex>;


const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 16.0;
override X : u32 = 64u;
override Y : u32 = 64u;
override Z : u32 = 64u;
// override NbVertices = 33u * 33u * 33u;

//const step : vec3f = vec3f

fn computeVertexId(globalId: u32) -> vec3u {
    var id = globalId;
    let z = id / ((X+1)*(Y+1));
    id %= ((X+1)*(Y+1));
    let y = id / (X+1);
    let x = id % (X+1);
    return vec3u(x, y, z);
}

fn computeVertexValue(pos: vec3f) -> f32 {
    // www-sop.inria.fr/galaad/surface/
    // chubs
    return pow(pos.x, 4)+pow(pos.y, 4)+pow(pos.z, 4)-pow(pos.x, 2)-pow(pos.y, 2)-pow(pos.z, 2)+0.5; 
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    _ = Z;
    if(global_id.x >= arrayLength(&vertices)) {
        return;
    }

    let vertexid = computeVertexId(global_id.x);
    var pos = vec3f(f32(vertexid.x), f32(vertexid.y), f32(vertexid.z));
    pos = pos * STEP + Min;
    let val = computeVertexValue(pos);


    vertices[global_id.x].position = pos;
    vertices[global_id.x].value = val;
    
    
    //position[global_id.x] *= STEP;
    //position[global_id.x] += Min;
    // position[global_id.x] = vec3f(f32(global_id.x));
}
