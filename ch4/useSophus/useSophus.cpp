#include <cmath>
#include <iostream>
using namespace std;

//#include <Eigen/Core>
//#include <Eigen/Geometry>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

using Sophus::SO3d;
using Sophus::SE3d;

int main(int argc, char** argv) {
  // 沿Z轴转90度的旋转矩阵
  Eigen::Matrix3d R =
      Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

  SO3d SO3_R(R);  // SO(3)可以直接从旋转矩阵构造
  SO3d SO3_v(SO3d::exp({0, 0, M_PI / 2}));  // 亦可从旋转向量构造
  Eigen::Quaterniond q(R);                  // 或者四元数
  SO3d SO3_q(q);
  // 上述表达方式都是等价的
  // 输出SO(3)时，以so(3)形式输出
  cout << "SO(3) from matrix: " << SO3_R.log() << endl;
  cout << "SO(3) from vector: " << SO3_v.log() << endl;
  cout << "SO(3) from quaternion :" << SO3_q.log() << endl;

  // 使用对数映射获得它的李代数
  Eigen::Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;
  // hat 为向量到反对称矩阵
  cout << "so3 hat=\n" << SO3d::hat(so3) << endl;
  // 相对的，vee为反对称到向
  cout << "so3 hat vee= " << SO3d::vee(SO3d::hat(so3)).transpose()
       << endl;  // transpose纯粹是为了输出美观一些

  // 增量扰动模型的更新
  Eigen::Vector3d update_so3(1e-4, 0, 0);  //假设更新量为这么多
  SO3d SO3_updated = SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated = " << SO3_updated.log() << endl;

  /********************萌萌的分割线*****************************/
  cout << "*************************" << endl;
  // 对SE(3)操作大同小异
  Eigen::Vector3d t(1, 0, 0);  // 沿X轴平移1
  SE3d SE3_Rt(R, t);           // 从R,t构造SE(3)
  SE3d SE3_qt(q, t);           // 从q,t构造SE(3)
  cout << "SE3 from R,t= " << endl << SE3_Rt.log() << endl;
  cout << "SE3 from q,t= " << endl << SE3_qt.log() << endl;
  // 李代数se(3) 是一个六维向量，方便起见先typedef一下
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.
  // 同样的，有hat和vee两个算符
  cout << "se3 hat = " << endl << SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << SE3d::vee(SE3d::hat(se3)).transpose() << endl;

  // 最后，演示一下更新
  Vector6d update_se3;  //更新量
  update_se3.setZero();
  update_se3(0, 0) = 1e-4d;
  SE3d SE3_updated = SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}
