#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/ray.h>
#include <nori/common.h>
#include <filesystem/resolver.h>
#include <sh/spherical_harmonics.h>
#include <sh/default_image.h>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <stb_image.h>
#include <type_traits>
#include <cassert>
#include <omp.h>

NORI_NAMESPACE_BEGIN

namespace ProjEnv {

	// 加载立方体贴图
	std::vector<std::unique_ptr<float[]>> LoadCubemapImages(const std::string& cubemapDir, int& width, int& height, int& channel) {
		// 定义立方体贴图六个面的文件名
		std::vector<std::string> cubemapNames{ "negx.jpg", "posx.jpg", "posy.jpg","negy.jpg", "posz.jpg", "negz.jpg" };

		// 存储六个面的图像数据
		std::vector<std::unique_ptr<float[]>> images(6);

		for (int i = 0; i < 6; i++) {
			// 构造每个面的文件路径
			std::string filename = cubemapDir + "/" + cubemapNames[i];
			int w, h, c;
			float* image = stbi_loadf(filename.c_str(), &w, &h, &c, 3);
			if (!image) {
				std::cout << "Failed to load image: " << filename << std::endl;
				exit(-1);
			}
			if (i == 0) {
				width = w;
				height = h;
				channel = c;
			} else if (w != width || h != height || c != channel) {
				std::cout << "Dismatch resolution for 6 images in cubemap" << std::endl;
				exit(-1);
			}
			images[i] = std::unique_ptr<float[]>(image);
			// int index = (0 * 128 + 0) * channel;
			// std::cout << images[i][index + 0] << "\t" << images[i][index + 1] << "\t"<< images[i][index + 2] << std::endl;
		}
		return images;
	}


	// 计算立方体贴图的立体角,被CalcArea调用
	float CalcPreArea(const float& x, const float& y) {
		//atan2的优势是可以正确处理异常的情况
		return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0));
	}

	float CalcArea(const float& u_, const float& v_, const int& width, const int& height) {
		//https://www.rorydriscoll.com/2012/01/15/cubemap-texel-solid-angle/
		// ( 0.5 is for texel center addressing)
		float u = (2.0 * (u_ + 0.5) / width) - 1.0;
		float v = (2.0 * (v_ + 0.5) / height) - 1.0;

		float invResolutionW = 1.0 / width;
		float invResolutionH = 1.0 / height;

		// u and v are the [-1,1] texture coordinate on the current face.
		// get projected area for this texel
		float x0 = u - invResolutionW;
		float y0 = v - invResolutionH;
		float x1 = u + invResolutionW;
		float y1 = v + invResolutionH;
		float angle = CalcPreArea(x0, y0) - CalcPreArea(x0, y1) -
			CalcPreArea(x1, y0) + CalcPreArea(x1, y1);

		return angle;
	}


	// 立方体贴图各个面的方向
	const Eigen::Vector3f cubemapFaceDirections[6][3] = {
	   {{0, 0, 1}, {0, -1, 0}, {-1, 0, 0}},  // negx
	   {{0, 0, 1}, {0, -1, 0}, {1, 0, 0}},   // posx
	   {{1, 0, 0}, {0, 0, -1}, {0, -1, 0}},  // negy
	   {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},    // posy
	   {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, // negz
	   {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},   // posz
	};

	//我需要实现的函数,预计算环境光的球谐函数
	template <size_t SHOrder>
	std::vector<Eigen::Array3f> PrecomputeCubemapSH(const std::vector<std::unique_ptr<float[]>>& images, const int& width, const int& height, const int& channel) {

		// 存储每个像素的方向向量
		std::vector<Eigen::Vector3f> cubemapDirs;

		// 预分配内存
		cubemapDirs.reserve(6 * width * height);

		//cubemap上的每个像素对应的方向
		for (int i = 0; i < 6; i++) {

			// 当前立方体贴图的面的 XYZ 轴方向
			Eigen::Vector3f faceDirX = cubemapFaceDirections[i][0];
			Eigen::Vector3f faceDirY = cubemapFaceDirections[i][1];
			Eigen::Vector3f faceDirZ = cubemapFaceDirections[i][2];

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {

					//back to ndc
					float u = 2 * ((x + 0.5) / width) - 1;
					float v = 2 * ((y + 0.5) / height) - 1;
					//计算出当前像素在三维空间中的方向向量 dir
					Eigen::Vector3f dir = (faceDirX * u + faceDirY * v + faceDirZ).normalized();
					cubemapDirs.push_back(dir);
				}
			}
		}

		//初始化ShCoeffiecents
		constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1);// 根据阶数计算球谐系数的总数量
		std::vector<Eigen::Array3f> SHCoeffiecents(SHNum);

		// 初始化球谐系数为零
		for (int i = 0; i < SHNum; i++)
			SHCoeffiecents[i] = Eigen::Array3f(0);

		for (int i = 0; i < 6; i++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {

					// TODO: here you need to compute light sh of each pixel of each face of cubemap 

					// 获取当前像素的方向向量
					Eigen::Vector3f dir = cubemapDirs[i * width * height + y * width + x];

					//像素的索引，channel是传入的参数，表示每个像素的通道数，通常是 3（表示 RGB）
					int index = (y * width + x) * channel;

					//RGB值,Le是从环境贴图取到的环境光照。图像数据通常以一维数组的形式存储，每个像素的 RGB 值连续排列
					Eigen::Array3f Le(images[i][index + 0], images[i][index + 1],
									  images[i][index + 2]);

					//计算 cubemap 上每个像素所代表的矩形区域投影到单位球面的面积
					auto delta_w = CalcArea(x, y, width, height);

					//SHOrder是球谐函数的阶数
					for (int l = 0; l <= SHOrder; l++) {
						//每一阶有2l+1个基函数，因此一共循环2l+1次
						for (int m = -l; m <= l; m++) {
							//计算基函数
							auto basic_sh_proj = sh::EvalSH(l, m, Eigen::Vector3d(dir.x(), dir.y(), dir.z()).normalized());
							//计算le在基函数上的投影，并且用黎曼和近似积分,投影到基函数上得到球谐系数的大小，因此光照部分的拟合结果是保存球谐系数li，而不是li*Bi
							//同样的，传输函数的拟合结果中，也是保存的球谐系数，不包含基函数，而基函数是单位正交的，不需要显式保存
							SHCoeffiecents[sh::GetIndex(l, m)] += Le * basic_sh_proj * delta_w;
						}
					}
				}
			}
		}
		return SHCoeffiecents;
	}


}

class PRTIntegrator : public Integrator {

public:
	static constexpr int SHOrder = 2;
	static constexpr int SHCoeffLength = (SHOrder + 1) * (SHOrder + 1);

	enum class Type {
		Unshadowed = 0,
		Shadowed = 1,
		Interreflection = 2
	};

	//构造函数
	PRTIntegrator(const PropertyList& props) {
		/* No parameters this time */
		m_SampleCount = props.getInteger("PRTSampleCount", 100);
		m_Bounce = props.getInteger("bounce", 1);
		m_CubemapPath = props.getString("cubemap");
		auto type = props.getString("type", "unshadowed");
		if (type == "unshadowed") {
			m_Type = Type::Unshadowed;
		} else if (type == "shadowed") {
			m_Type = Type::Shadowed;
		} else if (type == "interreflection") {
			m_Type = Type::Interreflection;
			m_Bounce = props.getInteger("bounce", 1);
		} else {
			throw NoriException("Unsupported type: %s.", type);
		}
	}

	//计算间接光照的球谐系数
	template<typename T>
	std::unique_ptr<std::vector<double>> computeInterreflectionSH(Eigen::MatrixXf* directTSHCoeffs, const Point3f& pos, const Normal3f& normal, T&& Lds, const Scene* scene, int bounces) {
		
		// 创建一个动态数组存储球谐系数
		std::unique_ptr<std::vector<double>> coeffs(new std::vector<double>());

		// 初始化所有系数为零
		coeffs->assign(SHCoeffLength, 0.0);

		// 遍历所有系数
		for (int i = 0; i < coeffs->size(); i++) {
			(*coeffs)[i] += Lds[i];// 累加直接光照的系数
		}

		// 如果反弹次数达到上限,则返回当前系数
		if (bounces >= m_Bounce)
			return coeffs;

		// 计算采样网格的边长
		const int sample_side = static_cast<int>(floor(sqrt(m_SampleCount)));

		// 遍历采样网格的行
		for (int t = 0; t < sample_side; t++) {
			// 遍历采样网格的列
			for (int p = 0; p < sample_side; p++) {

				// 生成随机采样点的 x 坐标
				double x1 = (t + nori::genRandomFloat()) / sample_side;
				// 生成随机采样点的 y 坐标
				double x2 = (p + nori::genRandomFloat()) / sample_side;

				// 计算采样方向的方位角
				double phi = 2.0 * M_PI * x1;
				// 计算采样方向的极角
				double theta = acos(2.0 * x2 - 1.0);

				// 将球坐标转换为方向向量
				Eigen::Array3d d = sh::ToVector(phi, theta);

				// 获取方向向量
				const auto wi = Vector3f(d.x(), d.y(), d.z());

				// 计算概率密度函数
				double pdf = 1.0 / (4 * M_PI);

				// 计算方向向量与法线的夹角余弦值
				double H = wi.normalized().dot(normal);

				// 定义一个交点对象
				Intersection its;
				// 如果方向有效且射线与场景相交
				if (H > 0.0 && scene->rayIntersect(Ray3f(pos, wi.normalized()), its)) {
					// 获取相交三角形的法线
					MatrixXf normals = its.mesh->getVertexNormals();
					// 获取相交三角形的索引
					Point3f idx = its.tri_index;
					// 获取相交点的位置
					Point3f hitPos = its.p;
					// 获取重心坐标
					Vector3f bary = its.bary;
					//利用重心坐标插值三角形各顶点的法向量,从而计算相交点的法线
					Normal3f hitNormal =
						Normal3f(normals.col(idx.x()).normalized() * bary.x() +
								 normals.col(idx.y()).normalized() * bary.y() +
								 normals.col(idx.z()).normalized() * bary.z())
						.normalized();

					//重心坐标插值三角形各顶点的(V * brdf * wiDotN)投影到球谐基函数后得到的coeffs值,即插值相交点的球谐系数
					auto interpolateSH =
						directTSHCoeffs->col(idx.x()) * bary.x() +
						directTSHCoeffs->col(idx.y()) * bary.y() +
						directTSHCoeffs->col(idx.z()) * bary.z();

					// 递归计算下一次反弹的系数
					auto nextBouncesCoeffs = computeInterreflectionSH(directTSHCoeffs, hitPos, hitNormal, interpolateSH, scene, bounces + 1);

					// 遍历所有系数
					for (int i = 0; i < SHCoeffLength; i++) {
						//采样到投影后的coeffes乘以cos做权重，这里不是蒙特卡洛积分。
						(*coeffs)[i] += (*nextBouncesCoeffs)[i] * H / m_SampleCount;
						// (*coeffs)[i] += 1 / M_PI * (*nextBouncesCoeffs)[i] * H / pdf / m_SampleCount;//Incorrect method
					}
				}
			}
		}

		return coeffs;
	}


	//我需要实现完整的函数
	//预处理阶段
	virtual void preprocess(const Scene* scene) override {
		
		// Here only compute one mesh,  获取场景中的第一个网格
		const auto mesh = scene->getMeshes()[0];

		// Projection environment, 解析立方体贴图路径
		auto cubePath = getFileResolver()->resolve(m_CubemapPath);

		// 生成环境光系数文件路径
		auto lightPath = cubePath / "light.txt";
		// 生成传输系数文件路径
		auto transPath = cubePath / "transport.txt";

		// 打开环境光系数文件
		std::ofstream lightFout(lightPath.str());
		// 打开传输系数文件
		std::ofstream fout(transPath.str());


		int width, height, channel;
		std::vector<std::unique_ptr<float[]>> images = ProjEnv::LoadCubemapImages(cubePath.str(), width, height, channel);

		//环境光的球谐系数
		auto envCoeffs = ProjEnv::PrecomputeCubemapSH<SHOrder>(images, width, height, channel);

		

		// 预计算得到的环境光系数保存在 m_LightCoeffs, 调整环境光系数矩阵的大小
		m_LightCoeffs.resize(3, SHCoeffLength);

		// 遍历环境光的所有球谐系数
		for (int i = 0; i < envCoeffs.size(); i++) {
			// 将系数写入文件
			//cubemap中每个像素le的值投影到球面谐波的基函数上得到coeffs
			lightFout << (envCoeffs)[i].x() << " " << (envCoeffs)[i].y() << " " << (envCoeffs)[i].z() << std::endl;
			//使用ProjEnv::PrecomputeCubemapSH预计算得到的环境光系数保存在 m_LightCoeffs 中
			m_LightCoeffs.col(i) = (envCoeffs)[i];
		}
		/*                    envCoeffs[0] , envCoeffs[1] ... envCoeffs[SHCoeffLength-1]
		 * m_LightCoeffs = [  envCoeffs[0] , envCoeffs[1] ... envCoeffs[SHCoeffLength-1] ]
		 *                    envCoeffs[0] , envCoeffs[1] ... envCoeffs[SHCoeffLength-1]
		*/
		std::cout << "Computed light sh coeffs from: " << cubePath.str() << " to: " << lightPath.str() << std::endl;


		// Projection transport , column-major
		// 调整传输系数矩阵的大小,这是一个(SHOrder + 1)^2 × VertexCount 的矩阵，每一列 j 存储了第 j 个顶点的所有球谐系数
		m_TransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount());

		// 写入顶点数量
		fout << mesh->getVertexCount() << std::endl;

		// 遍历所有顶点
		for (int i = 0; i < mesh->getVertexCount(); i++) {

			// 获取顶点位置
			const Point3f& v = mesh->getVertexPositions().col(i);

			// 获取顶点法线
			const Normal3f& n = mesh->getVertexNormals().col(i);

			//一个lambda函数, 定义传输函数:prt将渲染方程拆分为  光照  和  定义在表面的传输函数.对光照和传输函数进行球谐函数拟合
			//而std::vector<Eigen::Array3f> PrecomputeCubemapSH就是用来拟合光照的,接下来在拟合传输函数(三种情况)
			auto shFunc = [&](double phi, double theta) -> double {
				Eigen::Array3d d = sh::ToVector(phi, theta);
				const auto wi = Vector3f(d.x(), d.y(), d.z());

				double H = wi.normalized().dot(n.normalized());
				if (m_Type == Type::Unshadowed) {
					// TODO: here you need to calculate unshadowed transport term of a given direction
					// TODO: 此处你需要计算给定方向下的unshadowed传输项球谐函数值
					return H > 0.0 ? H : 0;
				} else {
					// TODO: here you need to calculate shadowed transport term of a given direction
					// TODO: 此处你需要计算给定方向下的shadowed传输项球谐函数值
					if (H > 0.0 && !scene->rayIntersect(Ray3f(v, wi.normalized()))) {
						return H;
					}
					return 0;
				}
				};

			// 将传输函数投影到球谐空间
			auto shCoeff = sh::ProjectFunction(SHOrder, shFunc, m_SampleCount);

			// 遍历所有系数, 将系数存储到成员变量中
			for (int j = 0; j < shCoeff->size(); j++) {

				m_TransportSHCoeffs.col(i).coeffRef(j) = (*shCoeff)[j];
			}

		}

		// 如果类型为间接反射,拟合
		if (m_Type == Type::Interreflection) {
			// TODO: leave for bonus
			

			// 调整间接传输系数矩阵的大小
			m_InterTransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount());

			// 使用 OpenMP 并行化
#pragma omp parallel for
			for (int i = 0; i < mesh->getVertexCount(); i++) { // 遍历所有顶点

				// 获取顶点位置
				const Point3f& v = mesh->getVertexPositions().col(i);

				// 获取顶点法线
				const Normal3f& n = mesh->getVertexNormals().col(i).normalized();

				// 计算间接光照的球谐系数
				auto indirectCoeffs = computeInterreflectionSH(&m_TransportSHCoeffs, v, n, m_TransportSHCoeffs.col(i), scene, 0);

				// 遍历所有系数
				for (int j = 0; j < SHCoeffLength; j++) {
					m_InterTransportSHCoeffs.col(i).coeffRef(j) = (*indirectCoeffs)[j];
				}
				std::cout << "computing interreflection light sh coeffs, current vertex idx: " << i << " total vertex idx: " << mesh->getVertexCount() << std::endl;
			}

			// 更新传输系数矩阵
			m_TransportSHCoeffs = m_InterTransportSHCoeffs;
		}

		// Save in face format
		// 遍历所有三角形
		for (int f = 0; f < mesh->getTriangleCount(); f++) {
			// 获取三角形索引
			const MatrixXu& F = mesh->getIndices();

			// 获取三角形的三个顶点索引
			uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);

			// 写入第一个顶点的系数
			for (int j = 0; j < SHCoeffLength; j++) {
				fout << m_TransportSHCoeffs.col(idx0).coeff(j) << " ";
			}
			fout << std::endl;

			// 写入第二个顶点的系数
			for (int j = 0; j < SHCoeffLength; j++) {
				fout << m_TransportSHCoeffs.col(idx1).coeff(j) << " ";
			}
			fout << std::endl;

			// 写入第三个顶点的系数
			for (int j = 0; j < SHCoeffLength; j++) {
				fout << m_TransportSHCoeffs.col(idx2).coeff(j) << " ";
			}
			fout << std::endl;

		}
		std::cout << "Computed SH coeffs"
			<< " to: " << transPath.str() << std::endl;
		//至此,编译运行 ,即可用球谐函数拟合指定环境光贴图以及光传输部分了
	}

	//渲染函数
	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {

		// 定义一个交点对象
		Intersection its;
		if (!scene->rayIntersect(ray, its))
			return Color3f(0.0f);

		
		const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> sh0 = m_TransportSHCoeffs.col(its.tri_index.x()),// 获取第一个顶点的传输系数
			sh1 = m_TransportSHCoeffs.col(its.tri_index.y()),// 获取第二个顶点的传输系数
			sh2 = m_TransportSHCoeffs.col(its.tri_index.z());// 获取第三个顶点的传输系数


		
		const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> rL = m_LightCoeffs.row(0), // 获取环境光的红色系数
			gL = m_LightCoeffs.row(1), // 获取环境光的绿色系数
			bL = m_LightCoeffs.row(2);// 获取环境光的蓝色系数

		//应用球谐函数拟合的系数计算光照
		Color3f c0 = Color3f(rL.dot(sh0), gL.dot(sh0), bL.dot(sh0)),// 计算第一个顶点的颜色
			c1 = Color3f(rL.dot(sh1), gL.dot(sh1), bL.dot(sh1)), // 计算第二个顶点的颜色
			c2 = Color3f(rL.dot(sh2), gL.dot(sh2), bL.dot(sh2));// 计算第三个顶点的颜色

		// 获取重心坐标
		const Vector3f& bary = its.bary;
		// 根据重心坐标插值最终颜色
		Color3f c = bary.x() * c0 + bary.y() * c1 + bary.z() * c2;
		return c;
	}

	std::string toString() const {
		return "PRTIntegrator[]";
	}

private:
	Type m_Type;
	int m_Bounce;
	int m_SampleCount;
	std::string m_CubemapPath;
	Eigen::MatrixXf m_TransportSHCoeffs;
	Eigen::MatrixXf m_InterTransportSHCoeffs;
	Eigen::MatrixXf m_LightCoeffs;
};

NORI_REGISTER_CLASS(PRTIntegrator, "prt");
NORI_NAMESPACE_END