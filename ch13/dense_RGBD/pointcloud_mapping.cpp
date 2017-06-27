#include <fstream>
#include <iostream>
using namespace std;

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  vector<cv::Mat> color_images, depth_images;
  vector<Eigen::Isometry3d> poses;  // camera poses

  ifstream fin("./data/pose.txt");
  if (!fin) {
    cerr << "cannot find pose file" << endl;
    return 1;
  }

  for (int i = 0; i < 5; i++) {
    boost::format fmt("./data/%s/%d.%s");
    color_images.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
    depth_images.push_back(
        cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));

    double data[7] = {0};
    for (int i = 0; i < 7; i++) {
      fin >> data[i];
    }
    Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
    poses.push_back(T);
  }

  // Camera intrinsics
  double cx = 325.5;
  double cy = 253.5;
  double fx = 518.0;
  double fy = 519.0;
  double depth_scale = 1000.0;

  cout << "Converting image to point cloud ..." << endl;

  typedef pcl::PointXYZRGB PointT;
  typedef pcl::PointCloud<PointT> PointCloud;

  // Create a new point cloud
  PointCloud::Ptr point_cloud(new PointCloud);
  for (int i = 0; i < 5; i++) {
    PointCloud::Ptr current(new PointCloud);
    cout << "转换图像中: " << i + 1 << endl;
    cv::Mat color = color_images[i];
    cv::Mat depth = depth_images[i];
    Eigen::Isometry3d T = poses[i];
    for (int v = 0; v < color.rows; v++)
      for (int u = 0; u < color.cols; u++) {
        unsigned int d = depth.ptr<unsigned short>(v)[u];  // 深度值
        if (d == 0) continue;     // 为0表示没有测量到
        if (d >= 7000) continue;  // 深度太大时不稳定，去掉
        Eigen::Vector3d point;
        point[2] = double(d) / depth_scale;
        point[0] = (u - cx) * point[2] / fx;
        point[1] = (v - cy) * point[2] / fy;
        Eigen::Vector3d point_world = T * point;

        PointT p;
        p.x = point_world[0];
        p.y = point_world[1];
        p.z = point_world[2];
        p.b = color.data[v * color.step + u * color.channels()];
        p.g = color.data[v * color.step + u * color.channels() + 1];
        p.r = color.data[v * color.step + u * color.channels() + 2];
        current->points.push_back(p);
      }
    // depth filter and statistical removal
    PointCloud::Ptr tmp(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> stats_filter;
    stats_filter.setMeanK(50);
    stats_filter.setStddevMulThresh(1.0);
    stats_filter.setInputCloud(current);
    stats_filter.filter(*tmp);
    (*point_cloud) += *tmp;
  }

  point_cloud->is_dense = false;
  cout << "There are " << point_cloud->size() << "points." << endl;

  // voxel filter
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setLeafSize(0.01, 0.01, 0.01);  // resolution
  PointCloud::Ptr tmp(new PointCloud);
  voxel_filter.setInputCloud(point_cloud);
  voxel_filter.filter(*tmp);
  tmp->swap(*point_cloud);

  cout << "After filtering, there are " << point_cloud->size() << "points."
       << endl;

  pcl::io::savePCDFileBinary("map.pcd", *point_cloud);
  return 0;
}
