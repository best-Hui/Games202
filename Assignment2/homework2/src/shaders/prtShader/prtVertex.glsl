attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;

attribute mat3 aPrecomputeLT;//这里保存的也是传输函数进行球谐函数拟合后的球谐系数，不包含也没必要包含球谐基函数

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

//rgb三个通道，而预计算中使用的是2阶球谐函数近似的，需要9个球谐系数，因此这里uPrecomputeL[i]是对于每一个通道使用3*3的矩阵来保存球谐系数
uniform mat3 uPrecomputeL[3];

varying highp vec3 vColor;

//这里的逻辑可以参考games202的课程6的1:21:27处，最终计算公式是光照的球谐系数乘以传输函数的球谐系数的累加，而无需显式乘以个基函数，因为基函数正交且归一，不同基函数乘积为0，相同的为1
float L_dot_LT(mat3 PrecomputeL, mat3 PrecomputeLT) {
  vec3 L_0 = PrecomputeL[0];
  vec3 L_1 = PrecomputeL[1];
  vec3 L_2 = PrecomputeL[2];
  vec3 LT_0 = PrecomputeLT[0];
  vec3 LT_1 = PrecomputeLT[1];
  vec3 LT_2 = PrecomputeLT[2];
  return dot(L_0, LT_0) + dot(L_1, LT_1) + dot(L_2, LT_2);//也就是这里实现的乘积结果的累加
}
void main(void) {
  //rgb
  for(int i = 0; i < 3; i++)
  {
    vColor[i] = L_dot_LT(uPrecomputeL[i],aPrecomputeLT);
  }

  gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aVertexPosition, 1.0);
}