
#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // d  = disparity value of pixel mn in image
            ushort d = depth.ptr<ushort>(m)[n];

            // if there is no disparity calculated for the pixel, continue
            if (d == 0)
                continue;
            // p is the 3d point
            PointT p;

            // apply convertion to 3d
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            // use rgb value
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // insert point in pointcloud
            cloud->points.push_back( p );
        }
    // cloud parameters
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

// apply transformation 2d for 3d in a single point
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}
