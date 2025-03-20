class Shader {

    constructor(gl, vsSrc, fsSrc, shaderLocations) {
        this.gl = gl;
        const vs = this.compileShader(vsSrc, gl.VERTEX_SHADER);
        const fs = this.compileShader(fsSrc, gl.FRAGMENT_SHADER);

        this.program = this.addShaderLocations({
            glShaderProgram: this.linkShader(vs, fs),
        }, shaderLocations);
    }

    compileShader(shaderSource, shaderType) {
        const gl = this.gl;
        var shader = gl.createShader(shaderType);
        gl.shaderSource(shader, shaderSource);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error(shaderSource);
            console.error('shader compiler error:\n' + gl.getShaderInfoLog(shader));
        }

        return shader;
    };

    linkShader(vs, fs) {
        const gl = this.gl;
        var prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);

        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            abort('shader linker error:\n' + gl.getProgramInfoLog(prog));
        }
        return prog;
    };

    
    // shaderLocations= {
    //         uniforms: this.#flatten_uniforms,
    //         attribs: this.#flatten_attribs
    //                  }

    // flatten_uniforms={'uPrecomputeL[0]': { type: 'precomputeL', value: null},
    // 'uPrecomputeL[1]': { type: 'precomputeL', value: null},
    // 'uPrecomputeL[2]': { type: 'precomputeL', value: null}
    // }

    // shaderLocations.uniforms[0] ='uPrecomputeL[0]'
    // shaderLocations.uniforms[1] ='uPrecomputeL[1]'等

    // 那么gl.getUniformLocation(result.glShaderProgram, shaderLocations.uniforms[0])相当于在result.glShaderProgram即着色器程序id中找到uPrecomputeL[0]的位置

    // Object.assign(target, ...sources);
    // target：目标对象，即要将属性复制到的对象。
    // sources：一个或多个源对象，它们的属性会被复制到目标对象中。
    // 返回值：返回目标对象（target），此时目标对象已经被修改。

    // Object.assign(result.uniforms, {
    //     [shaderLocations.uniforms[i]]: gl.getUniformLocation(result.glShaderProgram, shaderLocations.uniforms[i]),
    // });相当于:
    // let key = shaderLocations.uniforms[i];  // 假设是 "uPrecomputeL[0]"
    // let location = gl.getUniformLocation(result.glShaderProgram, key);
    
    // let obj = {
    //     [key]: location
    // };即：
    // 生成对象{
    //     "uPrecomputeL[0]": WebGLUniformLocation(...)
    // }
    // Object.assign 会将这个临时对象的属性复制到 result.uniforms 中。因此，代码的逻辑是：
    // 从 shaderLocations.uniforms 中取出键名（如 "uPrecomputeL[0]"）。
    // 使用 gl.getUniformLocation 查询该 uniform 的位置。
    // 创建一个临时对象，将查询到的位置赋值给对应的键名。
    // 将这个临时对象的属性复制到 result.uniforms 中。
    // 最终，result.uniforms 会变成：
    // {
    // "uPrecomputeL[0]": WebGLUniformLocation(...),
    // "uPrecomputeL[1]": WebGLUniformLocation(...)
    // ...
    // }
    // 也就是说，这样后续可以通过gl.uniform3fv(uniforms["uPrecomputeL[0]"], [1.0, 0.0, 0.0])来对着色器变量进行赋值，其中uniforms["uPrecomputeL[0]"]已经获取到了位置
    // 查询着色器程序中指定的 uniform 和 attribute 变量的位置，并将它们存储在 this.program 对象中；result是着色器程序id
    addShaderLocations(result, shaderLocations) {
        const gl = this.gl;
        result.uniforms = {};
        result.attribs = {};

        if (shaderLocations && shaderLocations.uniforms && shaderLocations.uniforms.length) {
            // 遍历 shaderLocations.uniforms 中的每个 uniform 变量名称
            for (let i = 0; i < shaderLocations.uniforms.length; ++i) {
                // 使用 gl.getUniformLocation(result.glShaderProgram, uniformName) 查询 uniform 变量的位置，并存储到 result.uniforms 中
                // 使用 Object.assign 将查询结果存储到 result.uniforms 中。键是变量名称（i），值是查询到的位置。
                result.uniforms = Object.assign(result.uniforms, {
                    [shaderLocations.uniforms[i]]: gl.getUniformLocation(result.glShaderProgram, shaderLocations.uniforms[i]),
                });
            }
        }
        if (shaderLocations && shaderLocations.attribs && shaderLocations.attribs.length) {
            for (let i = 0; i < shaderLocations.attribs.length; ++i) {
                result.attribs = Object.assign(result.attribs, {
                    [shaderLocations.attribs[i]]: gl.getAttribLocation(result.glShaderProgram, shaderLocations.attribs[i]),
                });
            }
        }

        return result;
    }
}
