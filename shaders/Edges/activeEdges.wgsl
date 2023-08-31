struct Vertex {
    position: vec3f,
    value: f32,
}

@group(0) @binding(0)
var <storage, read> vertices : array<Vertex>;
@group(0) @binding(1)
var <storage, read_write> edgesId : array<u32>;
@group(0) @binding(2)
var<storage, read_write> chunkEdges : array<u32>;

const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 16.0;
override X : u32 = 64u;
override Y : u32 = 64u;
override Z : u32 = 64u;
override NbVertices = (X+1) * (Y+1) * (Z+1);
override CHUNKSIZE = 64u;
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

@compute @workgroup_size(64)
fn markActiveEdges(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)) {
        return;
    }

    let cell = edgeCell(global_id.x);

    let axis = vec3u(
        u32(global_id.x < nbX),
        u32(global_id.x < nbY + nbX && global_id.x >= nbX),
        u32(global_id.x >= nbX + nbY)
    );

    let v0 = vertices[vertexId(cell)].value;
    let v1 = vertices[vertexId(cell+axis)].value;

    if((v0 * v1) <= 0.0) {
        edgesId[global_id.x] = global_id.x; 
    } else {
        edgesId[global_id.x] = 0u - 1u;
    }
}

@compute @workgroup_size(64)
fn countEdgesPerChunk(@builtin(global_invocation_id) global_id : vec3<u32>) {
    _ = X;
    _ = Y;
    _ = Z;

    /// first edge of chunk
    var offset = global_id.x * CHUNKSIZE;

    chunkEdges[global_id.x] = CHUNKSIZE;
}