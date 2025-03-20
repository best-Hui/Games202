class PRTMaterial extends Material {

    constructor(vertexShader, fragmentShader) {

        super({
            'uPrecomputeL[0]': { type: 'precomputeL', value: null},
            'uPrecomputeL[1]': { type: 'precomputeL', value: null},
            'uPrecomputeL[2]': { type: 'precomputeL', value: null},
        }, //uniforms参数
        ['aPrecomputeLT'], //attribs
        vertexShader, fragmentShader, null);
        //构造函数会将传入的uniforms参数列表逐个加入flatten_uniforms，将attribs加入flatten_attribs，在compile函数（编译着色器）时，
    }
}

// constructor(uniforms, attribs, vsSrc, fsSrc, frameBuffer) {
//     this.uniforms = uniforms;
//     this.attribs = attribs;
//     this.#vsSrc = vsSrc;
//     this.#fsSrc = fsSrc;
    
//     this.#flatten_uniforms = ['uViewMatrix','uModelMatrix', 'uProjectionMatrix', 'uCameraPos', 'uLightPos'];
//     for (let k in uniforms) {
//         this.#flatten_uniforms.push(k);
//     }
//     this.#flatten_attribs = attribs;

//     this.frameBuffer = frameBuffer;
// }

async function buildPRTMaterial(vertexPath, fragmentPath) {


    let vertexShader = await getShaderString(vertexPath);
    let fragmentShader = await getShaderString(fragmentPath);

    return new PRTMaterial(vertexShader, fragmentShader);

}