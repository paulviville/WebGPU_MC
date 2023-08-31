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
            // Z: Z,
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

const edgeOffStorageBuffer = device.createBuffer({
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
            // Z: Z,
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
    }],
});



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


passEncoder.end();

console.log(Math.ceil(nbVertices / 64))

commandEncoder.copyBufferToBuffer(
    vertexStorageBuffer,
    0, //offset
    vertexStagingBuffer,
    0, // offset
    nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT   
);

commandEncoder.copyBufferToBuffer(
    edgeIdStorageBuffer,
    0,
    edgeIdStagingBuffer,
    0,
    nbEdges * Uint32Array.BYTES_PER_ELEMENT,
)

commandEncoder.copyBufferToBuffer(
    chunkActiveEdgesStorageBuffer,
    0,
    chunkActiveEdgesStagingBuffer,
    0,
    (nbChunks +1) * Uint32Array.BYTES_PER_ELEMENT,
)

const commands = commandEncoder.finish();
device.queue.submit([commands]);

await vertexStagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, //offset
    nbVertices * 4 * Float32Array.BYTES_PER_ELEMENT
);



await edgeIdStagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, //offset
    nbEdges * Uint32Array.BYTES_PER_ELEMENT
);
const copyArrayBuffer = edgeIdStagingBuffer.getMappedRange(0, nbEdges*Uint32Array.BYTES_PER_ELEMENT);
const data = copyArrayBuffer.slice();
console.log(copyArrayBuffer.slice())
edgeIdStagingBuffer.unmap();
let t = [...(new Uint32Array(data))].filter(u => u != 4294967295)

console.log(t)


await chunkActiveEdgesStagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, //offset
    (nbChunks + 1) * Uint32Array.BYTES_PER_ELEMENT
);
const copyArrayBuffer2 = chunkActiveEdgesStagingBuffer.getMappedRange(0, (nbChunks+1)*Uint32Array.BYTES_PER_ELEMENT);
const data2 = copyArrayBuffer2.slice();
console.log(copyArrayBuffer2.slice())
edgeIdStagingBuffer.unmap();
let t2 = [...(new Uint32Array(data2))]//.filter(u => u != 4294967295)

console.log(t2)














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


let skip = false;
/// rendering
function render() {

    if(!skip){
    //move to render function
    const uniformArray = new Float32Array([
        ...camera.position, 0.0,
        ...camera.view,
        ...camera.projection,
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    camera.update();
    colorAttachment.view = context.getCurrentTexture().createView();
    const renderCommandEncoder = device.createCommandEncoder();
    const renderPass = renderCommandEncoder.beginRenderPass({
        colorAttachments: [colorAttachment],
        depthStencilAttachment: depthStencilAttachment,
    });
    renderPass.setPipeline(gridPipeline);
    renderPass.setBindGroup(0, gridBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, model.indexType);
    renderPass.drawIndexed(model.numIndices, nbVertices);
    renderPass.end();

    device.queue.submit([renderCommandEncoder.finish()]);
    }
    skip = !skip;
    requestAnimationFrame(render)
}
render();