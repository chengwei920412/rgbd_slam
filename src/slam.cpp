
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


int main( int argc, char** argv )
{
    // read image index limits form parameters file
    ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );

    // vector to store keyframes obtained
    vector< FRAME > keyframes;

    // initialize by reading first frame and camera parameters
    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex; // start by first index
    FRAME currFrame = readFrame( currIndex, pd ); // read first frame
    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    // compute keypoints and descriptors for the first frame
    computeKeyPointsAndDesp( currFrame, detector, descriptor );
    // create pointcloud for the first frame
    PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );


    // initialize slam solver using lm algorithm
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm( solver );
    globalOptimizer.setVerbose( false );


    // initalize vertex object
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex ); // vertex id = frame index
    v->setEstimate( Eigen::Isometry3d::Identity() ); // vertex transform = indentity
    v->setFixed( true ); // first position is fixed
    globalOptimizer.addVertex( v ); // add first vertex
    keyframes.push_back( currFrame ); // store fist frame

    // get keyframe minimum norm threshold and bool for check loop closure
    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");


    //start loop for adding keyframes
    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        // read a frame from image folder
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readFrame( currIndex,pd );

        // computeKeyPointsAndDesp for the frame
        computeKeyPointsAndDesp( currFrame, detector, descriptor );

        // get result of matching frames
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer );
        switch (result)
        {
        case NOT_MATCHED:
            cout<<RED"Not enough inliers."<<endl;
            break;

        case TOO_FAR_AWAY:
            cout<<RED"Too far away, may be an error."<<endl;
            break;

        case TOO_CLOSE:
            cout<<RESET"Too close, not a keyframe"<<endl;
            break;

        // DETECTED NEW KEYFRAME
        case KEYFRAME:
            cout<<GREEN"This is a new keyframe"<<endl;
            if (check_loop_closure)
            {   // check for loop closure
                checkNearbyLoops( keyframes, currFrame, globalOptimizer );
                checkRandomLoops( keyframes, currFrame, globalOptimizer );
            }
            // add frame to keyframes
            keyframes.push_back( currFrame );
            break;

        default:
            break;
        }
    }


    // OPTIMIZATION ============================================================

    cout<<RESET"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("./data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 500 ); // number of iterations
    globalOptimizer.save( "./data/result_after.g2o" ); // save graph
    cout<<"Optimization done."<<endl;


    // POINT CLOUD MAP GENERATION ==============================================

    cout<<"saving the point cloud map..."<<endl;
    PointCloud::Ptr output ( new PointCloud() ); // global map
    PointCloud::Ptr tmp ( new PointCloud() );

    pcl::VoxelGrid<PointT> voxel; // voxel grid filter for downsample
    pcl::PassThrough<PointT> pass; // z direction of pass filter
    pass.setFilterFieldName("z");
    pass.setFilterLimits( 0.0, 4.0 ); // limit filter for 4m

    // get the voxel grid size from parameters.txt
    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );

    for (size_t i=0; i<keyframes.size(); i++)
    {
        std::cout << "faltam: " << keyframes.size() - i << std::endl;

         // get the vertex from g2o, containing the optimized transformation
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿

        // create new pointcloud from frame
        PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //转成点云
        // filter pointcloud
        voxel.setInputCloud( newCloud );
        voxel.filter( *tmp );
        pass.setInputCloud( tmp );
        pass.filter( *newCloud );
        // apply transformation to the pointcloud to form the global map
        pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
        // add it to global map
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }

    voxel.setInputCloud( output );
    voxel.filter( *tmp );

    // save map pcd
    pcl::io::savePCDFile( "./data/result.pcd", *tmp );
    cout<<"Final map is saved."<<endl;
    //globalOptimizer.clear();

    return 0;
}
