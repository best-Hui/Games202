#version 300 es

#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWS;
uniform sampler2D uGShadow;
uniform vec3 uZBufferParams;

uniform sampler2D uDepthTexture[12];

in mat4 vWorldToScreen;
in vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

#define MAX_MIPMAP_LEVEL 9

out vec4 FragColor;

//随机数生成函数
float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

//随机数生成函数
vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

//随机数生成函数，根据片段坐标初始化随机数种子
float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

//半球面均匀采样函数，返回一个局部坐标系的位置
vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

//半球面基于余弦分布的采样，返回一个局部坐标系的位置
vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

//局部坐标系构建，根据法线n构建局部坐标系，生成两个正交基向量b1和b2
void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

// 将非线性深度值转换为线性深度值
float Linear01Depth( float z )
{
  float farDivNear = uZBufferParams.y / uZBufferParams.x;
  return 1.0 / ( ( 1.0 - farDivNear) * z + farDivNear);
}

//将齐次坐标投影到屏幕空间，透视除法
vec4 Project(vec4 a) {
  return a / a.w;
}

//计算世界空间点的线性深度值
float GetDepth(vec3 posWorld) {
  //screen space linearDepth
  float screenZ = Project(vWorldToScreen * vec4(posWorld, 1.0)).z * 0.5 + 0.5;
  screenZ = Linear01Depth(screenZ);
  return screenZ;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
 //将世界空间点转换为屏幕空间坐标
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

//从G-Buffer中读取深度
float GetGBufferDepth(vec2 uv) {
  return Linear01Depth(texture(uGDepth, uv).x);
}
//从G-Buffer中读取法线
vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture(uGNormalWS, uv).xyz;
  return normal;
}
//从G-Buffer中读取阴影
float GetGBufferuShadow(vec2 uv) {
  float visibility = texture(uGShadow, uv).x;
  return visibility;
}
//从G-Buffer中读取漫反射反照率
vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec2 uv) {
  vec3 albedo  = GetGBufferDiffuse(uv);
  return albedo * INV_PI;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
 //着色点位于 uv 处得到的光源的辐射度，并且需要考虑遮挡关系
vec3 EvalDirectionalLight(vec2 uv) {
  vec3 Le = GetGBufferuShadow(uv) * uLightRadiance;
  return Le;
}

// 根据BRDF的lobe，生成一根或多根光线，我们假设这里是镜面反射，那就只需要考虑一根反射光线，
//然后以固定步长沿着反射光线进行步进，每次步进都需要检查步进后光线的深度和场景的深度，
//直到光线深度大于等于场景深度，就获取该交点处的albedo，然后根据渲染方程进行Shading
bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  float step = 0.02;
  const int totalStepTimes = 1000; 
  int curStepTimes = 0;

  vec3 stepDir = normalize(dir) * step;
  vec3 curPos = ori;
  for(int curStepTimes = 0; curStepTimes < totalStepTimes; curStepTimes++)
  {
    float curDepth = GetDepth(curPos);
    vec2 curScreenUV = GetScreenCoordinate(curPos);
    if(curScreenUV.x < 0.0 || curScreenUV.x > 1.0 || curScreenUV.y < 0.0 || curScreenUV.y > 1.0)
      break;
    float gBufferDepth = GetGBufferDepth(curScreenUV);

    if(curDepth - gBufferDepth > 0.0001){
      hitPos = curPos;
      return true;
    }
    //o + t * d
    curPos += stepDir;
  }

  return false;
}

// test Screen Space Ray Tracing 
vec3 EvalReflect(vec3 wi, vec3 wo, vec2 uv) {
  vec3 worldNormal = GetGBufferNormalWorld(uv);
  vec3 relfectDir = normalize(reflect(-wo, worldNormal));
  vec3 hitPos;
  if(RayMarch(vPosWorld.xyz, relfectDir, hitPos)){
      vec2 screenUV = GetScreenCoordinate(hitPos);
      return GetGBufferDiffuse(screenUV);
  }
  else{
    return vec3(0.0);
  }
}

#define SAMPLE_NUM 1

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec3 L = vec3(0.0);

  vec3 worldPos = vPosWorld.xyz;
  vec2 screenUV = GetScreenCoordinate(vPosWorld.xyz);
  vec3 wi = normalize(uLightDir);
  vec3 wo = normalize(uCameraPos - worldPos);

  //test
  //L = EvalReflect(wi,wo,screenUV);

   //直接光照，直接光照的计算是基于物理模型的直接应用，不需要对采样过程进行校正，所以不需要除以 PDF。
   //L = V * Le * brdf * cos
   vec3 L_Normal = GetGBufferNormalWorld(screenUV);
   L = EvalDiffuse(screenUV) * EvalDirectionalLight(screenUV) * max(0., dot(L_Normal, wi));

  //间接光
  vec3 L_ind = vec3(0.0);
  // SAMPLE_NUM为1，因此有很多噪点
  for(int i = 0; i < SAMPLE_NUM; i++){
    float pdf;
    vec3 localDir = SampleHemisphereCos(s, pdf);
    vec3 L_ind_Normal = GetGBufferNormalWorld(screenUV);
    vec3 b1, b2;
    LocalBasis(L_ind_Normal, b1, b2);
    vec3 dir = normalize(mat3(b1, b2, L_ind_Normal) * localDir);

    //world space pos
    vec3 hitPos;
    if(RayMarch(worldPos, dir, hitPos)){
      vec2 hitScreenUV = GetScreenCoordinate(hitPos);
      //castRay =  V * Le * brdf * cos.
      vec3 hitNormal = GetGBufferNormalWorld(hitScreenUV);
      vec3 castRay = EvalDiffuse(hitScreenUV) * EvalDirectionalLight(hitScreenUV) * max(0., dot(hitNormal, wi));
      //L_ind += castRay * brdf * cos / pdf
      // 需要从半球面上随机采样多个方向，这些方向是根据某种分布生成的，因此需要除以 PDF 来校正采样偏差，确保结果是无偏的
      L_ind += castRay * EvalDiffuse(screenUV) * max(0., dot(L_ind_Normal, dir)) / pdf;
    }
  }
  L_ind /= float(SAMPLE_NUM);
  L = L + L_ind;

  vec3 color = pow(L, vec3(1.0 / 2.2));
  FragColor = vec4(vec3(color.rgb), 1.0);
}