#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/io/file_io.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>

int
main(int argc, char** argv)
{
	// Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the SHOT descriptors for each point.
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());

	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) != 0)
	{
		return -1;
		std::cout<<"FALSE"<<std::endl;
	}

	// Note: you would usually perform downsampling now. It has been omitted here
	// for simplicity, but be aware that computation can take a long time.

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(10);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// SHOT estimation object.
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	shot.setInputCloud(cloud);
	shot.setInputNormals(normals);
	// The radius that defines which of the keypoint's neighbors are described.
	// If too large, there may be clutter, and if too small, not enough points may be found.
	shot.setRadiusSearch(10);
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
