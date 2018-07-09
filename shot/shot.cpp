#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/keypoints/iss_3d.h>
// #include <pcl/features/flare.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/io/file_io.h>
#include <pcl/filters/uniform_sampling.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>

double
computeCloudResolution (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZ> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {

    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

int
main(int argc, char** argv)
{
	// Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames(new pcl::PointCloud<pcl::ReferenceFrame>());
	// Object for storing the SHOT descriptors for each point.
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZ> ());

	double resolution;

	// resolution=computeCloudResolution(cloud)

	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) != 0)
	{
		return -1;
		std::cout<<"FALSE"<<std::endl;
	}
	resolution=computeCloudResolution(cloud);
	// std::cout<<"res"<<resolution<<std::endl;
	// Estimate the normals.
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setKSearch(10);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
	// // 
	// pcl::ISSKeypoint3D<pcl::PointXYZ,pcl::PointXYZ> detector;
 //    detector.setInputCloud(cloud);
 //    detector.setSalientRadius(6 * resolution);
 //    detector.setNonMaxRadius(4* resolution);
 //    detector.setMinNeighbors(5);
 //    detector.setNormalRadius (4 * resolution);
	// detector.setBorderRadius (resolution);
 //    detector.setThreshold21(0.975);
 //    detector.setThreshold32(0.975);
 //    detector.setNumberOfThreads(4);
 //    detector.setSearchMethod(kdtree);
	// detector.compute(*keypoints);
	
	pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
  	uniform_sampling.setInputCloud (cloud);
  	uniform_sampling.setRadiusSearch (0.01);
  	uniform_sampling.filter (*keypoints);
  	// FRAMES
  	pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::ReferenceFrame> fr_est;
    fr_est.setRadiusSearch (1);
    fr_est.setSearchSurface (cloud);
    fr_est.setInputCloud(keypoints);
	fr_est.compute (*frames);
	// SHOT estimation object.
	// std::cout<<"lol"<<keypoints->points.size()<<std::endl;
	pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	shot.setInputCloud(keypoints);
	shot.setInputNormals(normals);
	shot.setSearchSurface (cloud);
	// The radius that defines which of the keypoint's neighbors are described.
	// // If too large, there may be clutter, and if too small, not enough points may be found.
	 shot.setSearchMethod(kdtree);
	shot.setRadiusSearch(1.0);
  	shot.setInputReferenceFrames(frames);
	shot.compute(*descriptors);
	std::cout << "SHOT output points.size (): " << descriptors->points.size () << std::endl;
	/*for(int i=0;i<descriptors->points.size ();i++)
	{
		std::cout << i << descriptors->points[i]<<std::endl;
	}*/
	//pcl::io::savePCDFile (const std::string &file_name, const pcl::PointCloud<PointT> &cloud, bool binary_mode = false)
	pcl::io::savePCDFile (argv[2], *descriptors, false);
	// pcl::visualization::PCLVisualizer viewer("PCL Viewer");
 //             viewer.setBackgroundColor (0.0, 0.0, 0.5);
 //             viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals);

 //             while (!viewer.wasStopped ())
 //             {
 //               viewer.spinOnce ();
 //             }
 //             return 0;
	}