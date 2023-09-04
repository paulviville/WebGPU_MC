import { Loader } from "./utils/loader.js";
import { Vertex } from "./utils/mesh.js";
import { Model } from "./utils/model.js";
import { OrbitCamera } from "./utils/orbit-camera.js";

console.log("WebGPU_MC")

// grid structure 
// grid order: along x, then y, then z;
//            6             7
//            +-------------+               +-----3-------+   
//          / |           / |             / |            /|   
//        /   |         /   |           6   10        7   11
//    4 +-----+-------+  5  |         +-----+2------+     |   
//      |   2 +-------+-----+ 3       |     +-----1-+-----+   
//      |   /         |   /           8   4         9   5
//      | /           | /             | /           | /       
//    0 +-------------+ 1             +------0------+         
//
// grid dimensions
const X = 16, Y = 16, Z = 16;
const nbVertices = (X+1) * (Y+1) * (Z+1);
const nbEdges = X*(Y+1)*(Z+1) + (X+1)*Y*(Z+1) + (X+1)*(Y+1)*Z;
const nbFaces = X*Y*(Z+1) + (X+1)*Y*Z + X*(Y+1)*Z;
const nbCubes = X*Y*Z;

console.log(`grid: ${X}x${Y}x${Z}
nb vertices: ${nbVertices}
nb edges: ${nbEdges}
nb faces: ${nbFaces}
nb cubes: ${nbCubes}
`)


// initializing webgpu
if(!navigator.gpu) throw Error("No GPU");

const adapter = await navigator.gpu.requestAdapter();
if(!adapter) throw Error("No adapter");

const device = await adapter.requestDevice();
if(!device) throw Error ("No device");

// console.log(device.limits)


// creating first compute 
const shaderCode = await fetch("./shaders/gridVertices.wgsl").then((response) => response.text())

const module = device.createShaderModule({
    label: 'first compute module',
    code: shaderCode,
});

const bindGroupLayout = device.createBindGroupLayout({
    label: 'first compute bindgroup layout',
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    }],
});

const pipeline = device.createComputePipeline({
    label: 'first compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
        module,
        entryPoint: "main",
        constants: {
            X: X,
            Y: Y,
            Z: Z,
        }
    },
});

const vertexStorageBuffer = device.createBuffer({
    label: "vertex position storage buffer",
    size: nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const vertexStagingBuffer = device.createBuffer({
    label: "vertex position staging buffer",
    size: nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const bindGroup = device.createBindGroup({
    label: 'first compute bind group',
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: {
            buffer: vertexStorageBuffer,
        },
    }],
});





const edgeComputeShaderCode = await fetch("./shaders/Edges/activeEdges.wgsl").then((response) => response.text())
const edgeComputemodule = device.createShaderModule({
    label: 'edge compute shader module',
    code: edgeComputeShaderCode,
});



const edgeIdStorageBuffer = device.createBuffer({
    label: "edge id storage buffer",
    size: nbEdges * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const edgeOffsetStorageBuffer = device.createBuffer({
    label: "edge chunk offset storage buffer",
    size: nbEdges * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const edgeIdStagingBuffer = device.createBuffer({
    label: "edge id staging buffer",
    size: nbEdges * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const chunkSize = 64
const nbChunks = Math.ceil(nbEdges / chunkSize);
const chunkActiveEdgesStorageBuffer = device.createBuffer({
    label: "nb edges/chunk storage buffer",
    size: (nbChunks+1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const chunkActiveEdgesStagingBuffer = device.createBuffer({
    label: "nb edges/chunk staging buffer",
    size: (nbChunks+1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const edgeMidStorageBuffer = device.createBuffer({
    label: "edge mid storage buffer",
    size: nbEdges * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
});

const edgeMidStagingBuffer = device.createBuffer({
    label: "edge mid staging buffer",
    size: nbEdges * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const rawTriStorageBuffer = device.createBuffer({
    label: "raw triangles storage buffer",
    size: nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const rawTriStagingBuffer = device.createBuffer({
    label: "raw triangles staging buffer",
    size: nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const cubeChunkSize = 64
const nbCubeChunks = Math.ceil(nbCubes / chunkSize);
const cubeCountChunkStorageBuffer = device.createBuffer({
    label: "nb edges/chunk storage buffer",
    size: (nbCubeChunks+1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const cubeCountChunkStagingBuffer = device.createBuffer({
    label: "nb edges/chunk staging buffer",
    size: (nbCubeChunks+1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});




const triIndexStorageBuffer = device.createBuffer({
    label: "sorted triangles storage buffer",
    size: nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDEX,
});

const triIndexStagingBuffer = device.createBuffer({
    label: "sorted staging buffer",
    size: nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});



const edgeComputeBindGroupLayout = device.createBindGroupLayout({
    label: 'edge compute bindgroup layout',
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'read-only-storage',
        },   
    },
    {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    },
    {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    },
    {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    },
    {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    },
    {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
            type: 'storage',
        },
    }],
});

const EdgeActiveComputePipeline = device.createComputePipeline({
    label: 'edge active compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "markActiveEdges",
        constants: {
            X: X,
            Y: Y,
            Z: Z,
        }
    },
});

const EdgePerChunkComputePipeline = device.createComputePipeline({
    label: 'edge per chunk compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "countEdgesPerChunk",
        constants: {
            X: X,
            Y: Y,
            Z: Z,
            CHUNKSIZE: chunkSize,
        }
    },
});

const chunkReducePipeline = device.createComputePipeline({
    label: 'edge per chunk compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "reduceEdgeCount",
        constants: {
            X: X,
            Y: Y,
            Z: Z,
            CHUNKSIZE: chunkSize,
        }
    },
});

const edgeOffsetPipeline = device.createComputePipeline({
    label: 'edge offset compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "completeEdgeOffsets",
        constants: {
            CHUNKSIZE: chunkSize,
        }
    },
});

const edgeMidPipeline = device.createComputePipeline({
    label: 'edge mid compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "computEdgeMid",

    },
});

const cellTrianglesComputePipeline = device.createComputePipeline({
    label: 'cube triangles compute pipeline',
    layout: device.createPipelineLayout({
        bindGroupLayouts: [edgeComputeBindGroupLayout],
    }),
    compute: {
        module: edgeComputemodule,
        entryPoint: "computeCellTriangles",
        constants: {
            X: X,
            Y: Y,
            Z: Z,
        }
    },
});
















const edgeIdComputeBindGroup = device.createBindGroup({
    label: 'edge id compute bind group',
    layout: edgeComputeBindGroupLayout,
    entries: [{
        binding: 0,
        resource: {
            buffer: vertexStorageBuffer,
        },
    },
    {
        binding: 1,
        resource: {
            buffer: edgeIdStorageBuffer,
        },
    },
    {
        binding: 2,
        resource: {
            buffer: chunkActiveEdgesStorageBuffer,
        },
    },
    {
        binding: 3,
        resource: {
            buffer: edgeOffsetStorageBuffer,
        },
    },
    {
        binding: 4,
        resource: {
            buffer: edgeMidStorageBuffer,
        },
    },
    {
        binding: 5,
        resource: {
            buffer: rawTriStorageBuffer,
        },
    }],
});


let p0 = performance.now();

const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();

passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(nbVertices / 64));


passEncoder.setPipeline(EdgeActiveComputePipeline);
passEncoder.setBindGroup(0, edgeIdComputeBindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(nbEdges / 64));

passEncoder.setPipeline(EdgePerChunkComputePipeline);
passEncoder.dispatchWorkgroups(Math.ceil(nbChunks / 64));

passEncoder.setPipeline(chunkReducePipeline);
passEncoder.dispatchWorkgroups(1);

passEncoder.setPipeline(edgeOffsetPipeline);
passEncoder.dispatchWorkgroups(Math.ceil(nbEdges / 64));

passEncoder.setPipeline(edgeMidPipeline);
passEncoder.dispatchWorkgroups(Math.ceil(nbEdges / 64));

passEncoder.setPipeline(cellTrianglesComputePipeline);
passEncoder.dispatchWorkgroups(Math.ceil(nbCubes / 64));


passEncoder.end();


// commandEncoder.copyBufferToBuffer(
//     vertexStorageBuffer,
//     0, //offset
//     vertexStagingBuffer,
//     0, // offset
//     nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT   
// );

// commandEncoder.copyBufferToBuffer(
//     edgeOffsetStorageBuffer,
//     0,
//     edgeIdStagingBuffer,
//     0,
//     nbEdges * Uint32Array.BYTES_PER_ELEMENT,
// )

commandEncoder.copyBufferToBuffer(
    chunkActiveEdgesStorageBuffer,
    0,
    chunkActiveEdgesStagingBuffer,
    0,
    (nbChunks +1) * Uint32Array.BYTES_PER_ELEMENT,   
)

// commandEncoder.copyBufferToBuffer(
//     edgeMidStorageBuffer,
//     0, //offset
//     edgeMidStagingBuffer,
//     0, // offset
//     nbEdges * 4 * Float32Array.BYTES_PER_ELEMENT   
// );

commandEncoder.copyBufferToBuffer(
    rawTriStorageBuffer,
    0,
    rawTriStagingBuffer,
    0,
    nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT,   
)


const commands = commandEncoder.finish();
device.queue.submit([commands]);
let p1 = performance.now();

console.log(p0, p1, p1 - p0);
// await vertexStagingBuffer.mapAsync(
//     GPUMapMode.READ,
//     0, //offset
//     nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT
// );

// const copyArrayBuffer0 = vertexStagingBuffer.getMappedRange(0, nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT);
// const vertexData = copyArrayBuffer0.slice();
// vertexStagingBuffer.unmap();
// let vertexArray = [...(new Float32Array(vertexData))];
// console.log(vertexArray)

// await edgeIdStagingBuffer.mapAsync(
//     GPUMapMode.READ,
//     0, //offset
//     nbEdges * Uint32Array.BYTES_PER_ELEMENT
// );
// const copyArrayBuffer = edgeIdStagingBuffer.getMappedRange(0, nbEdges*Uint32Array.BYTES_PER_ELEMENT);
// const data = copyArrayBuffer.slice();
// console.log(copyArrayBuffer.slice())
// edgeIdStagingBuffer.unmap();
// let t = [...(new Uint32Array(data))].filter(u => u != 4294967295)

// console.log(t)


// await chunkActiveEdgesStagingBuffer.mapAsync(
//     GPUMapMode.READ,
//     0, //offset
//     (nbChunks + 1) * Uint32Array.BYTES_PER_ELEMENT
// );
// const copyArrayBuffer2 = chunkActiveEdgesStagingBuffer.getMappedRange(0, (nbChunks+1)*Uint32Array.BYTES_PER_ELEMENT);
// const data2 = copyArrayBuffer2.slice();
// console.log(copyArrayBuffer2.slice())
// chunkActiveEdgesStagingBuffer.unmap();
// let t2 = [...(new Uint32Array(data2))]//.filter(u => u != 4294967295)

// // console.log(t2)



await rawTriStagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, //offset
    nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT
);
const copyArrayBufferCubes = rawTriStagingBuffer.getMappedRange(0, nbCubes * 15 * Uint32Array.BYTES_PER_ELEMENT);
const dataCubes = copyArrayBufferCubes.slice();
console.log(copyArrayBufferCubes.slice())
rawTriStagingBuffer.unmap();
let tCubes = [...(new Uint32Array(dataCubes))]//.filter(u => u != 4294967295)

console.log(tCubes)










let activeEdges = [];

function vertex(x, y, z){
    let v = x + (X+1) * y + (X+1)*(Y+1) * z
    return v;
    }

const nbX = (X)*(Y+1)*(Z+1);
const nbY = (X+1)*(Y)*(Z+1);
const nbZ = (X+1)*(Y+1)*(Z);
function edgeCoord(e) {
    const A = (e >= nbX) * nbX + (e >= (nbX + nbY)) * nbY;
    const B = (X + (e > nbX)) * (Y + ((e < nbX) || (e > nbX + nbY)));
    const C = X + (e > nbX)


    const Ex = 1*(e < nbX);
    const Ey = 1*(e < nbY + nbX && e >= nbX);
    const Ez = 1*(e >= nbX + nbY);

    let e_ = e - A;
    let z = Math.floor(e_ / B)
    e_ = e_ % B;
    let y = Math.floor(e_ / C);
    let x = e_ % C;

    // console.log(e)
    // console.log(x,y,z, vertex(x,y,z));
    // console.log(x+Ex,y+Ey,z+Ez, vertex(x+Ex,y+Ey,z+Ez));
    return {x, y, z};
}

function edgeVertices(e) {
    let eCoord0 = edgeCoord(e);
    let eCoord1 = edgeCoord(e);

    if(e < nbX)
        eCoord1.x += 1;
    else if(e < nbY + nbX)
        eCoord1.y += 1;
    else
        eCoord1.z += 1;

    let v0 =vertex(eCoord0.x, eCoord0.y, eCoord0.z);
    let v1 =vertex(eCoord1.x, eCoord1.y, eCoord1.z);

    v0 = vertexArray[v0 + 3];
    v1 = vertexArray[v1 + 3];
    if(v0 * v1 < 0)
        activeEdges.push({e, v0, v1});
    // console.log(vertex(eCoord0.x, eCoord0.y, eCoord0.z), vertex(eCoord1.x, eCoord1.y, eCoord1.z))
}

// for(let i = 0; i < nbEdges; ++i) 
//     edgeVertices(i)

//     console.log(activeEdges)


// await edgeMidStagingBuffer.mapAsync(
//     GPUMapMode.READ,
//     0, //offset
//     nbEdges * 4* Float32Array.BYTES_PER_ELEMENT
// );

// const copyMidArrayBuffer = edgeMidStagingBuffer.getMappedRange(0, nbEdges * 4* Float32Array.BYTES_PER_ELEMENT);
// const dataMid = copyMidArrayBuffer.slice();
// console.log(copyMidArrayBuffer.slice())
// edgeMidStagingBuffer.unmap();
// let tMid = [...(new Float32Array(dataMid))]//.filter(u => u != 0)

// console.log(tMid)







const canvas = document.getElementById('webGpuCanvas');
const context = canvas.getContext('webgpu');
context.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
});


const camera = new OrbitCamera(canvas);

/// create spheres
const assetLoader = new Loader;
const model = new Model(await assetLoader.loadModel('utils/billboard.json'));
const vertexBuffer = model.createVertexBuffer(device);
const indexBuffer = model.createIndexBuffer(device);


// create rendering resources
const uniformBuffer = device.createBuffer({
    label: 'sphere uniform buffer',
    size: 144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const depthTexture = device.createTexture({
    label: 'depth texture',
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});



const edgeLinesCode = await fetch("./shaders/Edges/edgeLines.wgsl").then((response) => response.text())
const edgeLinesShaderModule = device.createShaderModule({
    label: 'edge lines shader',
    code: edgeLinesCode,
});

const edgeLinesBindGroupLayout = device.createBindGroupLayout({
    label: 'edge lines bind group layout',
    entries: [
        { /// uniform buffer
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {},
        },
        { /// vertex storage buffer
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {
                type: 'read-only-storage',
            }
        },
        { /// edge id storage buffer
            binding: 2,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {
                type: 'read-only-storage',
            }
        }
    ]
});
const edgeLinesPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
        edgeLinesBindGroupLayout,
    ],
});

const edgeLinesPipeline = device.createRenderPipeline({
    label: 'edgeLines pipeline',
    layout: edgeLinesPipelineLayout,
    vertex: {
        module: edgeLinesShaderModule,
        entryPoint: 'vertex',
    },
    fragment: {
        module: edgeLinesShaderModule,
        entryPoint: 'fragment',
        targets: [{
            format: navigator.gpu.getPreferredCanvasFormat(),
        }]
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
    },
    primitive: {
        cullMode: 'back',
        topology: "line-strip"
    },
});

// const edgeLinesVertexArray = new Float32Array([
//     -1.0, -1.0, 0.0, 0.0,
//     1.0, 1.0, 0.0, 0.0,
//     0.0, 1.0, 0.0, 0.0
// ]);

// const edgeLinesVertexBuffer = device.createBuffer({
//     label: "edge lines vertex buffer placeholder",
//     size: edgeLinesVertexArray.byteLength,
//     usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
// });
// device.queue.writeBuffer(edgeLinesVertexBuffer, 0, edgeLinesVertexArray)

const edgeLinesBindGroup = device.createBindGroup({
    label: 'edgeLines bind group',
    layout: edgeLinesBindGroupLayout,
    entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: vertexStorageBuffer}},
        {binding: 2, resource: {buffer: edgeIdStorageBuffer}},
    ],
});

const gridShaderCode = await fetch("./shaders/Vertices/vertexBillboard.wgsl").then((response) => response.text())
const gridShaderModule = device.createShaderModule({
    label: 'grid vertex shader',
    code: gridShaderCode,
});

const gridBindGroupLayout = device.createBindGroupLayout({
    label: 'grid bind group layout',
    entries: [
        { /// uniform buffer
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {},
        },
        { /// vertex storage buffer
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {
                type: 'read-only-storage',
            }
        }
    ]
});

const gridPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
        gridBindGroupLayout,
    ],
});

const gridPipeline = device.createRenderPipeline({
    label: 'grid pipeline',
    layout: gridPipelineLayout,
    vertex: {
        module: gridShaderModule,
        entryPoint: 'vertex',
        buffers: [Vertex.vertexLayout()],
    },
    fragment: {
        module: gridShaderModule,
        entryPoint: 'fragment',
        targets: [{
            format: navigator.gpu.getPreferredCanvasFormat(),
        }]
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
    },
    primitive: {
        cullMode: 'back',
        // topology: "point-list"
    },
});

const gridBindGroup = device.createBindGroup({
    label: 'grid bind group',
    layout: gridBindGroupLayout,
    entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: vertexStorageBuffer}},
    ],
});

const edgeMidBindGroup = device.createBindGroup({
    label: 'edge mid bind group',
    layout: gridBindGroupLayout,
    entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: edgeMidStorageBuffer}},
    ],
});




const colorAttachment = {
    view: null,
    clearValue: {r: 0.8, g: 0.8, b: 0.8, a: 1},
    loadOp: 'clear',
    loadValue: {r: 0, g: 0, b: 0, a: 1},
    storeOp: 'store',
};

const depthStencilAttachment = {
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: 'discard',
};


/// rendering
function render() {

    //move to render function
    const uniformArray = new Float32Array([
        ...camera.position, 0.0,
        ...camera.view,
        ...camera.projection,
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // camera.update();
    colorAttachment.view = context.getCurrentTexture().createView();
    const renderCommandEncoder = device.createCommandEncoder();
    const renderPass = renderCommandEncoder.beginRenderPass({
        colorAttachments: [colorAttachment],
        depthStencilAttachment: depthStencilAttachment,
    });
    renderPass.setPipeline(gridPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, model.indexType);
    // renderPass.setBindGroup(0, gridBindGroup);
    // renderPass.drawIndexed(model.numIndices, nbVertices);

    renderPass.setBindGroup(0, edgeMidBindGroup);
    // renderPass.drawIndexed(model.numIndices, 6528);
    renderPass.drawIndexed(model.numIndices, 1440);





    // renderPass.setPipeline(edgeLinesPipeline);
    // renderPass.setBindGroup(0, edgeLinesBindGroup);
    // renderPass.draw(2, nbEdges);
    renderPass.end();

    device.queue.submit([renderCommandEncoder.finish()]);
    // requestAnimationFrame(render)
}
render();

function loop () {
    if(camera.update()) {
        render();
    }
    requestAnimationFrame(loop);
}

loop()