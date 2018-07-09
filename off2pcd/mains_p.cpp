#include <iostream>
#include <exception>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include "include/offparser.hpp"
#include "include/cg.hpp"
#include "include/writter.hpp"

#define DEFAULT_STEP_SIZE 0.1

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> CloudT;


double minimum(double x1, double x2, double x3)
{
  double min = (x1 < x2) ? x1 : x2;
  min = (min < x3) ? min : x3;
  return min;
}

double maximum(double x1, double x2, double x3)
{
  double max = (x1 > x2) ? x1 : x2;
  max = (max > x3) ? max : x3;
  return max;
}

void transposeToMassCenter( CloudT::Ptr & cloud )
{
  pcl::PointXYZ centroid;
  pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
        new  pcl::PointCloud<pcl::PointXYZ>);
  pcl::computeCentroid<pcl::PointXYZ, pcl::PointXYZ>(*cloud, centroid);

  for(pcl::PointXYZ & p : cloud->points)
  {
    p.x = p.x - centroid.x;
    p.y = p.y - centroid.y;
    p.z = p.z - centroid.z;
  }
  return ;
}

void normalizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
{
  pcl::PointXYZ p_max;
  pcl::PointXYZ p_min;
  pcl::getMinMax3D(*cloud, p_min, p_max);
  double max = std::abs(maximum(p_max.x, p_max.y, p_max.z));
  double min = std::abs(minimum(p_min.x, p_min.y, p_min.z));
  max = (min > max) ? min : max;
  for(pcl::PointXYZ & p : cloud->points)
  {
    p.x = p.x/max;
    p.y = p.y/max;
    p.z = p.z/max;
  }
  return ;
}

std::string OutPath(std::string file_path)
{
  int ind = file_path.find_last_of(".");
  return file_path.substr(0, ind+1) + "pcd";
}

int off2pcd(std::string file_path = "")
{
  // > Variables
  CloudT::Ptr cloud(new CloudT);
  CloudT::Ptr cloud_filtered(new CloudT);
  std::vector<cg::vertex> vertices;
  std::vector<cg::facade> facades;

  // > Parameters
  std::string out_path = OutPath(file_path);
  double step_size = DEFAULT_STEP_SIZE;
  bool visualize = false;

  // > Flow Control
  bool error_params = false;

  

  if(file_path.compare("") == 0)
  {
    std::cout << "> Argument missing: -p <FILE PATH>." << std::endl;
    error_params = true;
  }

  if(step_size <= 0 || step_size >= 1)
  {
    std::cout << "> Invalid argument: <STEP SIZE> must be greater than zero " <<
        "and smaller than one." << std::endl;
    error_params = true;
  }

  if(error_params)
  {
    std::cout << "> Detected errors. Program stopped." << std::endl;
    return -1;
  }

  if(utils::loadOFFFile(file_path, vertices, facades) != 0)
  {
    std::cout << "Error loading the file " << file_path << " ." << std::endl;
    return -1;
  }
  try
  {
    cg::Mesh mesh(vertices, facades);
    mesh.normalize();
    mesh.to_cloud(cloud, step_size);

    std::cout << "Input cloud size: " << cloud->size() << std::endl;

    utils::write_pcd(out_path, cloud);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "> Error: Arguments expected." << std::endl;
    std::cout << "> Possible arguments:" << std::endl;
    std::cout << "'-p dataset_path'" << std::endl;
    std::cout << "'-s <STEP SIZE>' (default is " <<
        DEFAULT_STEP_SIZE << ")" << std::endl;
    return -1;
  }

  // > Parameters reading
  int count = 1;
  std::string dataset_path = "";
  float step_size = DEFAULT_STEP_SIZE;
  while(count < argc)
  {
    // > Argument type
    std::string param = std::string(argv[count++]);
    // > File path
    if(!param.compare("-p"))
      dataset_path = std::string(argv[count++]);
    // > Step size parameter
    else if(!param.compare("-s"))
      step_size = atof(argv[count++]);
    else
    {
      std::cout << "=> Warning: argument " << param <<
          " is not a default argument" << std::endl;
      count++;
    }
  }

  std::ifstream fin(dataset_path + "/models.in");
  std::string file_path = "";
  while(fin >> file_path)
    off2pcd(file_path);
  fin.close();
  return 0;
}
