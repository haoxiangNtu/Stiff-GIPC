//
// gl_main.cpp
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
// #include "GIPC.cuh"
#include "device_launch_parameters.h"
#include "mlbvh.cuh"
#include <stdio.h>
#include "load_mesh.h"
#include "cuda_tools/cuda_tools.h"
#include <queue>
//#include "timer.h"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"
#include <gipc/type_define.h>
#include <filesystem>
#include <gipc/statistics.h>
#include <gipc/utils/simple_scene_importer.h>
#include <gipc/utils/urdf_scene_importer.h>
#include <Eigen/Geometry>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <GIPC.cuh>
#include "abd_system/abd_system.h"

// ImGui for interactive UI (joint control sliders)
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

auto             assets_dir = std::string{gipc::assets_dir()};
std::string      metis_dir  = assets_dir + "sorted_mesh/";

// Generate an NxN grid cloth OBJ at runtime. Returns the file path.
// Vertices span [-0.5, 0.5] in X and Z, Y=0.
static std::string generate_cloth_obj(int n)
{
    std::string path = assets_dir + "triMesh/cloth_" + std::to_string(n) + "x" + std::to_string(n) + ".obj";
    std::ofstream ofs(path);
    for(int j = 0; j <= n; j++)
        for(int i = 0; i <= n; i++)
            ofs << "v " << (double(i) / n - 0.5) << " 0 " << (double(j) / n - 0.5) << "\n";
    for(int j = 0; j < n; j++)
        for(int i = 0; i < n; i++)
        {
            int v0 = j * (n + 1) + i + 1;
            ofs << "f " << v0 << " " << (v0 + n + 1) << " " << (v0 + 1) << "\n";
            ofs << "f " << (v0 + 1) << " " << (v0 + n + 1) << " " << (v0 + n + 2) << "\n";
        }
    ofs.close();
    std::cout << "[generate_cloth] " << n << "x" << n
              << " -> " << ((n+1)*(n+1)) << " verts, " << (n*n*2) << " faces: " << path << std::endl;
    return path;
}
double           collision_detection_buff_scale = 1;
double           motion_rate                    = 1;
bool             g_skip_rendering               = false;
bool             g_headless_benchmark           = false;
int              g_headless_max_steps           = 500;
int              g_scene_no                     = 5;
double           linear_system_buff_scale       = 1.0;
mesh_obj         obj;
lbvh_f           bvh_f;
lbvh_e           bvh_e;
GIPC             ipc;
device_TetraData d_tetMesh;
tetrahedra_obj   tetMesh;
vector<Node>     nodes;
vector<AABB>     bvs;
vector<string>   obj_pathes;

// Whether interactive joint control UI is enabled (set by case 12)
bool g_joint_control_enabled = false;

// ---------------------------------------------------------------------------
// Trajectory playback infrastructure
// ---------------------------------------------------------------------------
struct TrajectoryKeyframe
{
    double              time;
    std::vector<double> revolute_angles;   // radians, one per revolute joint
    std::vector<double> prismatic_dists;   // metres, one per prismatic joint
};

static std::vector<TrajectoryKeyframe> g_trajectory;
static bool   g_trajectory_playback_enabled = false;
static double g_trajectory_sim_time         = 0.0;
static int    g_settling_frames             = 0;

static bool load_trajectory(const std::string& path)
{
    g_trajectory.clear();
    std::ifstream ifs(path);
    if(!ifs.is_open())
    {
        std::cerr << "[trajectory] Cannot open " << path << std::endl;
        return false;
    }

    std::string line;
    int n_revolute  = -1;
    int n_prismatic = -1;

    while(std::getline(ifs, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        std::istringstream ss(line);
        TrajectoryKeyframe kf;
        ss >> kf.time;
        double v;
        std::vector<double> vals;
        while(ss >> v)
            vals.push_back(v);

        if(n_revolute < 0)
        {
            n_revolute  = static_cast<int>(tetMesh.joint_angle_controls.size());
            n_prismatic = static_cast<int>(tetMesh.prismatic_drive_controls.size());
        }

        int total = n_revolute + n_prismatic;
        if(static_cast<int>(vals.size()) < total)
        {
            std::cerr << "[trajectory] Line has " << vals.size()
                      << " values, expected " << total << std::endl;
            continue;
        }
        kf.revolute_angles.assign(vals.begin(), vals.begin() + n_revolute);
        kf.prismatic_dists.assign(vals.begin() + n_revolute,
                                  vals.begin() + n_revolute + n_prismatic);
        g_trajectory.push_back(std::move(kf));
    }
    ifs.close();
    std::cout << "[trajectory] Loaded " << g_trajectory.size()
              << " keyframes from " << path
              << " (revolute=" << n_revolute << ", prismatic=" << n_prismatic << ")"
              << std::endl;
    return !g_trajectory.empty();
}

static void update_trajectory(double dt)
{
    if(g_trajectory.empty())
        return;

    g_trajectory_sim_time += dt;

    // Find the two bracketing keyframes for linear interpolation
    double t = g_trajectory_sim_time;
    if(t <= g_trajectory.front().time)
    {
        auto& kf = g_trajectory.front();
        for(size_t i = 0; i < kf.revolute_angles.size() && i < tetMesh.joint_angle_controls.size(); i++)
            tetMesh.joint_angle_controls[i].target_angle = kf.revolute_angles[i];
        for(size_t i = 0; i < kf.prismatic_dists.size() && i < tetMesh.prismatic_drive_controls.size(); i++)
            tetMesh.prismatic_drive_controls[i].target_distance = kf.prismatic_dists[i];
        return;
    }
    if(t >= g_trajectory.back().time)
    {
        auto& kf = g_trajectory.back();
        for(size_t i = 0; i < kf.revolute_angles.size() && i < tetMesh.joint_angle_controls.size(); i++)
            tetMesh.joint_angle_controls[i].target_angle = kf.revolute_angles[i];
        for(size_t i = 0; i < kf.prismatic_dists.size() && i < tetMesh.prismatic_drive_controls.size(); i++)
            tetMesh.prismatic_drive_controls[i].target_distance = kf.prismatic_dists[i];
        return;
    }

    // Binary search for the interval
    size_t lo = 0, hi = g_trajectory.size() - 1;
    while(lo + 1 < hi)
    {
        size_t mid = (lo + hi) / 2;
        if(g_trajectory[mid].time <= t)
            lo = mid;
        else
            hi = mid;
    }

    auto& kf0 = g_trajectory[lo];
    auto& kf1 = g_trajectory[hi];
    double alpha = (t - kf0.time) / (kf1.time - kf0.time + 1e-12);
    alpha = std::max(0.0, std::min(1.0, alpha));

    for(size_t i = 0; i < kf0.revolute_angles.size() && i < tetMesh.joint_angle_controls.size(); i++)
        tetMesh.joint_angle_controls[i].target_angle =
            kf0.revolute_angles[i] * (1.0 - alpha) + kf1.revolute_angles[i] * alpha;

    for(size_t i = 0; i < kf0.prismatic_dists.size() && i < tetMesh.prismatic_drive_controls.size(); i++)
        tetMesh.prismatic_drive_controls[i].target_distance =
            kf0.prismatic_dists[i] * (1.0 - alpha) + kf1.prismatic_dists[i] * alpha;
}
// ---------------------------------------------------------------------------
int              initPath = 0;
using namespace std;
int   step      = 0;
int   frameId   = 0;
int   surfNumId = 0;
float xRot      = 0.0f;
float yRot      = 0.f;
float xTrans    = 0;
float yTrans    = 0;
float zTrans    = 0;
int   ox;
int   oy;
int   buttonState;
float xRotLength    = 0.0f;
float yRotLength    = 0.0f;
float window_width  = 1000;
float window_height = 1000;
int   s_dimention   = 3;
bool  saveSurface   = true;
bool  change        = false;
bool  screenshot    = false;

bool isSetShader = false;

bool drawbvh     = false;
bool drawSurface = true;

bool stop = true;

double3 center;
double3 Ssize;

GLuint PN_vbo_;
GLuint VAO;
GLuint color_vbo_;
//GLuint color_vao_;
GLuint normal_vbo_;
//GLuint normal_vao_;
GLuint v;
GLuint f;
GLuint shaderProgram;

int            clothFaceOffset = 0;
int            bodyVertOffset  = 0;
double         global_offset   = 1.0;
vector<string> files;
vector<int>    file_vert_offsets;
vector<int>    file_tet_offsets;

void Init_CUDA()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }
}

#pragma pack(push, 1)
typedef struct
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} mBITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} mBITMAPINFOHEADER;
#pragma pack(pop)

bool WriteBitmapFile(int width, int height, const std::string& file_name, unsigned char* bitmapData)
{
    mBITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(mBITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(mBITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;  //BM
    bitmapFileHeader.bfOffBits = sizeof(mBITMAPFILEHEADER) + sizeof(mBITMAPINFOHEADER);

    mBITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(mBITMAPINFOHEADER));
    bitmapInfoHeader.biSize        = sizeof(mBITMAPINFOHEADER);
    bitmapInfoHeader.biWidth       = width;
    bitmapInfoHeader.biHeight      = height;
    bitmapInfoHeader.biPlanes      = 1;
    bitmapInfoHeader.biBitCount    = 24;
    bitmapInfoHeader.biCompression = 0L;
    bitmapInfoHeader.biSizeImage   = width * abs(height) * 3;

    //////////////////////////////////////////////////////////////////////////
    FILE*         filePtr;
    unsigned char tempRGB;
    int           imageIdx;

    for(imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB                  = bitmapData[imageIdx];
        bitmapData[imageIdx]     = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    filePtr = fopen(file_name.c_str(), "wb");
    if(NULL == filePtr)
    {
        return false;
    }

    fwrite(&bitmapFileHeader, sizeof(mBITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(mBITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
    return true;
}

void SaveScreenShot(int width, int height, const std::string& file_name)
{
    int   data_len    = height * width * 3;  // bytes
    void* screen_data = malloc(data_len);
    memset(screen_data, 0, data_len);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);
    WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);
    free(screen_data);
}

void saveSurfaceMesh(const string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId++);
    ss << ".obj";
    std::string file_path = ss.str();

    const size_t nSurfVerts = tetMesh.surfVerts.size();
    const size_t nFaces     = tetMesh.surface.size();

    std::vector<int> globalToLocal(tetMesh.vertexNum, -1);
    for(size_t i = 0; i < nSurfVerts; i++)
        globalToLocal[tetMesh.surfVerts[i]] = static_cast<int>(i);

    std::string buf;
    buf.reserve(nSurfVerts * 60 + nFaces * 40 + 64);

    char line[128];
    buf.append("s 1\n");
    for(size_t i = 0; i < nSurfVerts; i++)
    {
        const auto& pos = tetMesh.vertexes[tetMesh.surfVerts[i]];
        int len = snprintf(line, sizeof(line), "v %.8g %.8g %.8g\n", pos.x, pos.y, pos.z);
        buf.append(line, len);
    }

    for(size_t i = 0; i < nFaces; i++)
    {
        const auto& tri = tetMesh.surface[i];
        int len = snprintf(line, sizeof(line), "f %d %d %d\n",
                           globalToLocal[tri.x] + 1,
                           globalToLocal[tri.y] + 1,
                           globalToLocal[tri.z] + 1);
        buf.append(line, len);
    }

    FILE* fp = fopen(file_path.c_str(), "wb");
    if(fp)
    {
        fwrite(buf.data(), 1, buf.size(), fp);
        fclose(fp);
    }
}


void saveTets(const string& path)
{
    int tetIdoffset = 0;
    for(int ii = 0; ii < 4096; ii++)
    {
        //tetMesh.output_tetrahedraMesh
        std::stringstream ss;
        ss << path;
        ss << ii;  // / 10;
        //if (surfNumId % 10 != 0) return;
        ss << ".msh";
        std::string file_path = ss.str();
        ofstream    outmsh1(file_path);

        map<int, int> meshToSurf;
        //outSurf << "s 1" << endl;
        outmsh1 << "$Nodes\n";
        outmsh1 << file_vert_offsets[ii + 1] - file_vert_offsets[ii] << endl;
        for(int i = 0; i < file_vert_offsets[ii + 1] - file_vert_offsets[ii]; i++)
        {
            const auto& pos = tetMesh.vertexes[i + file_vert_offsets[ii]];
            outmsh1 << i + 1 << " " << pos.x << " " << pos.y << " " << pos.z << endl;
            meshToSurf[i + file_vert_offsets[ii]] = i;
        }
        outmsh1 << "$Elements\n";
        outmsh1 << file_tet_offsets[ii + 1] << endl;

        for(int i = 0; i < file_tet_offsets[ii + 1]; i++)
        {
            int tetId = i + tetIdoffset;
            outmsh1 << i + 1 << " 4 0 " << meshToSurf[tetMesh.tetrahedras[tetId].x] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].y] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].z] + 1 << " "
                    << meshToSurf[tetMesh.tetrahedras[tetId].w] + 1 << endl;
        }
        tetIdoffset += file_tet_offsets[ii + 1];
        outmsh1.close();
    }
}

void draw_box2D(float ox, float oy, float width, float height)
{
    glLineWidth(2.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox + width, oy, 0);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox, oy + height, 0);

    glVertex3f(ox + width, oy, 0);
    glVertex3f(ox + width, oy + height, 0);

    glVertex3f(ox + width, oy + height, 0);
    glVertex3f(ox, oy + height, 0);

    glEnd();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length, int boxType = 0)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.1f);
    if(boxType == 1)
    {
        glLineWidth(1.5f);
        glColor3f(0.8f, 0.8f, 0.8f);
    }
    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

void draw_lines(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);
    int numbers = 20;
    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox + width * i / numbers, oy, 0);
        glVertex3f(ox + width * i / numbers, oy + height, 0);
    }

    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy + height * i / numbers, 0);
        glVertex3f(ox + width, oy + height * i / numbers, 0);
    }

    glEnd();


    glLineWidth(1.5f);
    glColor3f(0.8f, 0.8f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(ox + width / 2, oy, 0);
    glVertex3f(ox + width / 2, oy + height, 0);

    glVertex3f(ox, oy + height / 2, 0);
    glVertex3f(ox + width, oy + height / 2, 0);

    glEnd();
}

void draw_mesh3D()
{
    glEnable(GL_DEPTH_TEST);
    glLineWidth(1.5f);
    glColor3f(0.9f, 0.1f, 0.1f);
    const vector<uint3>& surf = tetMesh.surface;  //obj.faces;
    glBegin(GL_TRIANGLES);


    for(int j = 0; j < tetMesh.surface.size(); j++)
    {
        glVertex3f((tetMesh.vertexes[surf[j].x].x),
                   (tetMesh.vertexes[surf[j].x].y),
                   (tetMesh.vertexes[surf[j].x].z));
        glVertex3f((tetMesh.vertexes[surf[j].y].x),
                   (tetMesh.vertexes[surf[j].y].y),
                   (tetMesh.vertexes[surf[j].y].z));
        glVertex3f((tetMesh.vertexes[surf[j].z].x),
                   (tetMesh.vertexes[surf[j].z].y),
                   (tetMesh.vertexes[surf[j].z].z));
    }
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    //glDisable(GL_DEPTH_TEST);
    glLineWidth(0.1f);
    glBegin(GL_LINES);

    for(int j = 0; j < tetMesh.surfEdges.size(); j++)
    {
        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].x].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].z));
        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].y].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].z));

        glColor3f(0.9f, 0.9f, 0.9f);
        glLineWidth(0.1f);
    }
    glEnd();

    // -- Visualize stitch springs as bright green lines --
    if (!d_tetMesh.stitch_paired_vertex.empty() && tetMesh.softNum > 0) {
        glDisable(GL_DEPTH_TEST);  // draw on top
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        for (int i = 0; i < tetMesh.softNum; ++i) {
            int fem_idx = tetMesh.targetIndex[i];
            int abd_idx = d_tetMesh.stitch_paired_vertex[i];
            if (abd_idx < 0) continue;
            glColor3f(0.0f, 1.0f, 0.0f);  // bright green
            glVertex3f(tetMesh.vertexes[fem_idx].x,
                       tetMesh.vertexes[fem_idx].y,
                       tetMesh.vertexes[fem_idx].z);
            glVertex3f(tetMesh.vertexes[abd_idx].x,
                       tetMesh.vertexes[abd_idx].y,
                       tetMesh.vertexes[abd_idx].z);
        }
        glEnd();
        glEnable(GL_DEPTH_TEST);
    }

    //glColor3f(0.99f, 0.1f, 0.1f);
    ////glDisable(GL_DEPTH_TEST);
    //glPointSize(8);
    //glBegin(GL_POINTS);
    //glVertex3f((tetMesh.vertexes[2189].x), (tetMesh.vertexes[2189].y), (tetMesh.vertexes[2189].z));
    //glColor3f(0.99f, 0.99f, 0.1f);
    //glVertex3f((tetMesh.vertexes[870].x), (tetMesh.vertexes[870].y), (tetMesh.vertexes[870].z));
    //glVertex3f((tetMesh.vertexes[905].x), (tetMesh.vertexes[905].y), (tetMesh.vertexes[905].z));
    //glVertex3f((tetMesh.vertexes[965].x), (tetMesh.vertexes[965].y), (tetMesh.vertexes[965].z));
    //glEnd();
}

void draw_bvh()
{
    int num = (bvs.size() + 1) / 2;
    for(int j = 0; j < bvs.size(); j++)
    {
        int   i = j;
        float ox, oy, oz, bwidth, bheight, blength;
        ox      = (bvs[i].lower.x);
        oy      = (bvs[i].lower.y);
        oz      = (bvs[i].lower.z);
        bwidth  = (bvs[i].upper.x - bvs[i].lower.x);
        bheight = (bvs[i].upper.y - bvs[i].lower.y);
        blength = (bvs[i].upper.z - bvs[i].lower.z);
        draw_box3D(ox, oy, oz, bwidth, bheight, blength);
    }
}

int            counttt = 0;
vector<float3> getRenderGeometry(int& number)
{

    vector<double3> meshNormal(tetMesh.vertexNum, make_double3(0, 0, 0));
    number = tetMesh.surface.size();  //meshTemp.surfaceRender.size();
    vector<float3> pos_normal_color(3 * number * 3);

    for(int i = 0; i < number; i++)
    {
        //int tetId = meshTemp.surfaceRender[i][3];
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        double3 vt0 = tetMesh.vertexes[v0];  // Vector3d(meshTemp.vertexes[v0][0], meshTemp.vertexes[v0][1], meshTemp.vertexes[v0][2]);
        double3 vt1 = tetMesh.vertexes[v1];  // Vector3d(meshTemp.vertexes[v1][0], meshTemp.vertexes[v1][1], meshTemp.vertexes[v1][2]);
        double3 vt2 = tetMesh.vertexes[v2];  // Vector3d(meshTemp.vertexes[v2][0], meshTemp.vertexes[v2][1], meshTemp.vertexes[v2][2]);
        double3 vec1 = __GEIGEN__::__minus(vt1, vt0);  //vt1 - vt0;
        double3 vec2 = __GEIGEN__::__minus(vt2, vt0);
        double3 normal =
            __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(vec1, vec2));  //vec1.cross(vec2).normalized();

        pos_normal_color[i * 9]     = make_float3(vt0.x, vt0.y, vt0.z);
        pos_normal_color[i * 9 + 3] = make_float3(vt1.x, vt1.y, vt1.z);
        pos_normal_color[i * 9 + 6] = make_float3(vt2.x, vt2.y, vt2.z);

        pos_normal_color[i * 9 + 2] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 5] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 8] = make_float3(0.6875f, 0.51953f, 0.38671f);
        //}


        meshNormal[v0] = __GEIGEN__::__add(meshNormal[v0], normal);  //normal;
        meshNormal[v1] = __GEIGEN__::__add(meshNormal[v1], normal);
        meshNormal[v2] = __GEIGEN__::__add(meshNormal[v2], normal);
    }
    for(int i = 0; i < number; i++)

    {
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        //meshNormal[v0].normalize(); meshNormal[v1].normalize(); meshNormal[v2].normalize();
        pos_normal_color[i * 9 + 1] =
            make_float3(meshNormal[v0].x, meshNormal[v0].y, meshNormal[v0].z);
        pos_normal_color[i * 9 + 4] =
            make_float3(meshNormal[v1].x, meshNormal[v1].y, meshNormal[v1].z);
        pos_normal_color[i * 9 + 7] =
            make_float3(meshNormal[v2].x, meshNormal[v2].y, meshNormal[v2].z);
    }

    return pos_normal_color;
}


void draw_Scene3D()
{
    //face.mesh3Ds[0] = mesh3d;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(!g_skip_rendering)
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glTranslatef(xTrans, yTrans, zTrans);
        glRotatef(xRot, 1.0f, 0.0f, 0.0f);
        glRotatef(yRot, 0.0f, 1.0f, 0.0f);

        //draw_box3D(-2, -1, -2, 4, 4, 4, 1);
        if(drawSurface)
        {
            draw_mesh3D();
        }
        if(drawbvh)
        {
            draw_bvh();
        }

        glPopMatrix();
    }

    // ---- ImGui Rendering ----
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::NewFrame();

    if(g_joint_control_enabled
       && (!tetMesh.joint_angle_controls.empty() || !tetMesh.prismatic_drive_controls.empty()))
    {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Joint State Control"))
        {
            if(!tetMesh.joint_angle_controls.empty())
            {
                ImGui::Text("Revolute Joints");
                ImGui::Separator();
                for(auto& ctrl : tetMesh.joint_angle_controls)
                {
                    float angle_deg = static_cast<float>(ctrl.target_angle * 180.0 / 3.14159265358979);
                    float lo_deg    = static_cast<float>(ctrl.lower_limit * 180.0 / 3.14159265358979);
                    float hi_deg    = static_cast<float>(ctrl.upper_limit * 180.0 / 3.14159265358979);
                    if(ImGui::SliderFloat(ctrl.joint_name.c_str(), &angle_deg, lo_deg, hi_deg, "%.1f"))
                    {
                        ctrl.target_angle = static_cast<double>(angle_deg) * 3.14159265358979 / 180.0;
                    }
                }
            }

            if(!tetMesh.prismatic_drive_controls.empty())
            {
                ImGui::Spacing();
                ImGui::Text("Prismatic Joints");
                ImGui::Separator();
                for(auto& pctrl : tetMesh.prismatic_drive_controls)
                {
                    float dist_mm = static_cast<float>(pctrl.target_distance * 1000.0);
                    float lo_mm   = static_cast<float>(pctrl.lower_limit * 1000.0);
                    float hi_mm   = static_cast<float>(pctrl.upper_limit * 1000.0);
                    if(ImGui::SliderFloat(pctrl.joint_name.c_str(), &dist_mm, lo_mm, hi_mm, "%.1f mm"))
                    {
                        pctrl.target_distance = static_cast<double>(dist_mm) / 1000.0;
                    }
                }
            }

            if(ImGui::Button("Reset All"))
            {
                for(auto& ctrl : tetMesh.joint_angle_controls)
                    ctrl.target_angle = 0.0;
                for(auto& pctrl : tetMesh.prismatic_drive_controls)
                    pctrl.target_distance = 0.0;
            }
        }
        ImGui::End();
    }

    // Info overlay
    {
        ImGui::SetNextWindowPos(ImVec2(10, static_cast<float>(window_height) - 60.0f), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.35f);
        if(ImGui::Begin("##info", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
            | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
        {
            ImGui::Text("Step: %d  |  %s", step, stop ? "PAUSED (Space to run)" : "RUNNING");
        }
        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    glutSwapBuffers();
    //glFlush();
}
double mfsum                   = 0;
double total_time              = 0;
int    total_cg_iterations     = 0;
int    total_newton_iterations = 0;
int    start                   = -1;

void saveScreenPic(const string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << step;
    std::string file_path = ss.str();

    SaveScreenShot(window_width, window_height, file_path);
}

void initFEM(tetrahedra_obj& mesh)
{

    double massSum   = 0;
    double volumeSum = 0;
    //float  angleX = FEM::PI / 4, angleY = -FEM::PI / 4, angleZ = FEM::PI / 2;
    //__GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
    //__GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
    //__GEIGEN__::__set_Mat_val(
    //    rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));


    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));

    ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));

    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    printf("ipc.shearStiff: %f\n", ipc.shearStiff);


    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);  //calculateDms3D_double(mesh.vertexes, mesh.tetrahedras[i], 0);

        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);

        double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        double lengthRateLame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volumeRateLame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        double lengthRate = 4 * lengthRateLame / 3;
        double volumeRate = volumeRateLame + 5 * lengthRateLame / 6;

        mesh.lengthRate.push_back(lengthRate);
        mesh.volumeRate.push_back(volumeRate);
    }

    for(int i = 0; i < mesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], DM);

        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);

        double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= ipc.clothThickness;
        mesh.area.push_back(area);


        mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

        massSum += area * ipc.clothDensity;
        volumeSum += area;
        mesh.tri_DM_inverse.push_back(DM_inverse);
    }

    mesh.meanMass = massSum / mesh.vertexNum;
    printf("meanMass: %f\n", mesh.meanMass);
    mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void DefaultSettings()
{
    // global settings
    ipc.density        = 1e3;
    ipc.PoissonRate    = 0.49;
    //ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
    //                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    //ipc.lengthRate        = 4 * ipc.lengthRateLame / 3;
    //ipc.volumeRate        = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.frictionRate      = 0.4;
    ipc.gd_frictionRate   = 0.4;
    ipc.clothThickness    = 1e-3;
    ipc.clothYoungModulus = 1e6;
    ipc.bendYoungModulus  = 1e5;
    //ipc.stretchStiff      = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.shearStiff        = ipc.stretchStiff * 0.3;
    ipc.clothDensity      = 2e2;
    ipc.strainRate        = 100;
    ipc.softMotionRate    = 1e0;
    ipc.bendStiff         = 3e-4;
    ipc.Newton_solver_threshold = 1e-2;
    ipc.pcg_threshold           = 1e-4;
    ipc.IPC_dt                  = 1e-2;
    ipc.relative_dhat           = 1e-3;
    //ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
    //                / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    //ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;
}
//int  meshids = 0;
void LoadSettings()
{
    bool successfulRead = false;

    //read file
    std::ifstream infile;


    string DEFAULT_CONFIG_FILE = std::string{gipc::assets_dir()} + "scene/parameterSetting.txt";


    infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
    if(successfulRead = infile.is_open())
    {
        int  tempEnum;
        char ignoreToken[256];

        // global settings:
        infile >> ignoreToken >> ipc.density;
        infile >> ignoreToken >> ipc.PoissonRate;
        infile >> ignoreToken >> ipc.frictionRate;
        infile >> ignoreToken >> ipc.gd_frictionRate;
        infile >> ignoreToken >> ipc.clothThickness;
        infile >> ignoreToken >> ipc.clothYoungModulus;
        infile >> ignoreToken >> ipc.bendYoungModulus;
        //infile >> ignoreToken >> ipc.shearStiff;
        infile >> ignoreToken >> ipc.clothDensity;
        infile >> ignoreToken >> ipc.strainRate;
        infile >> ignoreToken >> ipc.softMotionRate;
        //infile >> ignoreToken >> ipc.bendStiff;
        infile >> ignoreToken >> collision_detection_buff_scale;
        infile >> ignoreToken >> motion_rate;
        infile >> ignoreToken >> ipc.IPC_dt;
        infile >> ignoreToken >> ipc.pcg_threshold;
        infile >> ignoreToken >> ipc.Newton_solver_threshold;
        infile >> ignoreToken >> ipc.relative_dhat;
        //infile >> ignoreToken >> meshids;



        //ipc.shearStiff =
        infile.close();
    }

    if(!successfulRead)
    {
        std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
        DefaultSettings();
    }
}

void set_case1()
{
    double                    dist       = 0.2;
    int                       count      = 4;
    int                       count_Y    = 4;
    double                    fem_height = -0.8;
    double                    abd_height = -0.6;
    gipc::SimpleSceneImporter importer;

    linear_system_buff_scale = 1.0;

    double Youngth_Modulus = 1e4;
    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -abd_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::ABD,
                                       transform,
                                       1e5,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -fem_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::FEM,
                                       transform,
                                       Youngth_Modulus,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }
}


void set_case2()
{
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.2;
    double3                   position_offset = make_double3(0, -0.5, 0);
    Eigen::Matrix4d           transform       = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    linear_system_buff_scale = 1.0;
    double Youngth_Modulus = 1e4;
    string mesh0_path      = assets_dir + "tetMesh/bunny2.msh";
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::ABD,
                           transform,
                           Youngth_Modulus,
                           mesh0_path,
                           ipc.pcg_data.P_type);

    position_offset             = make_double3(0, 0.65, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    string mesh1_path = mesh0_path;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh1_path,
                           ipc.pcg_data.P_type);


    position_offset             = make_double3(0, 0, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1.0;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);
    string mesh2_path = assets_dir + "triMesh/cloth_high.obj";

    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh2_path,
                           ipc.pcg_data.P_type);
}

void set_case3()
{

    gipc::SimpleSceneImporter importer{assets_dir + "scene/json/wrecking-ball-simple.json",
                                       assets_dir + "tetMesh/wrecking-ball-mesh/",
                                       gipc::BodyType::ABD};
    linear_system_buff_scale = 1.0;
    importer.import_scene(tetMesh);
}

void set_case4()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.6;
    double3                   position_offset = make_double3(0, 1.0, 0);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(Eigen::Vector3d{0, 1.0, 0});
    t.scale(scale);
    t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));
    Eigen::Matrix4d transform = t.matrix();

    string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "fixed vertex num: " << fixed_vertex_num << std::endl;
}

void set_case5()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 2.0;
    gipc::SimpleSceneImporter importer;
    double                    scale = 1.0;
    Eigen::Vector3d           position_offset{0, 0, 0};

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(position_offset);
    t.scale(scale);
    Eigen::Matrix4d transform = t.matrix();

    string mesh_path       = assets_dir + "tetMesh/high_mat.msh";
    double Youngth_Modulus = 1e4;
    ipc.PoissonRate        = 0.48;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh_path,
                           ipc.pcg_data.P_type);

    // no gravity
    for(int i = 0; i < tetMesh.vertexes.size(); i++)
    {
        tetMesh.apply_gravity[i] = 0;
    }

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -0.5 + eps || tetMesh.vertexes[i].x > 0.5 - eps)
        {
            tetMesh.targetIndex.push_back(i);
            tetMesh.targetPos.push_back(tetMesh.vertexes[i]);
        }
    }
    tetMesh.softNum = tetMesh.targetIndex.size();
    std::cout << "soft constraint num: " << tetMesh.softNum << std::endl;
    ipc.softMotionRate = 1;

    const double angular_vel = 3.14159265358979323846/5;
    d_tetMesh.update_soft_constraint_functor =
        [angular_vel](double3 vertex, int step_id, double ipc_dt) -> double3
    {
        double3 rotated_vertex = vertex;
        if(vertex.x < 0)
        {
            // rotate along x axis clockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(angular_vel * ipc_dt)
                                  - vertex.z * std::sin(angular_vel * ipc_dt),
                              vertex.y * std::sin(angular_vel * ipc_dt)
                                  + vertex.z * std::cos(angular_vel * ipc_dt)};
        }
        if(vertex.x > 0)
        {
            // rotate along x axis counterclockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(-angular_vel * ipc_dt)
                                  - vertex.z * std::sin(-angular_vel * ipc_dt),
                              vertex.y * std::sin(-angular_vel * ipc_dt)
                                  + vertex.z * std::cos(-angular_vel * ipc_dt)};
        }
        return rotated_vertex;
    };
}

void set_case6()
{
    linear_system_buff_scale = 2.0;
    ipc.pcg_data.P_type = 1;
    double scale      = 0.3;
    double dist       = scale / 2;
    int    count      = 8;
    int    count_Y    = 15;
    double fem_height = global_offset + 1 - 0.8;
    double abd_height = fem_height - dist;


    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;


                double3 position_offset =
                    double3{-pos.x(), -abd_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            1e6,
                                            gipc::BodyType::ABD);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -fem_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            5e4,
                                            gipc::BodyType::ABD);
            }
        }
    }
    
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.scale(1.5);
    t.translate(Eigen::Vector3d(0, 0.35, 0));
    string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           t.matrix(),
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -1.5 + eps || tetMesh.vertexes[i].x > 1.5 - eps)
        {
            tetMesh.boundaryTypies[i] = 1;
        }
    }

    ipc.relative_dhat = 1e-3;
    ipc.strainRate    = 1e6;
}

// ==========================================================================
// Case 7: URDF import test
// Loads a simple test robot URDF with 3 cube links as ABD bodies.
// ==========================================================================
void set_case7_urdf_test()
{
    gipc::UrdfSceneImporter urdf_importer;

    // Path to the test URDF
    std::string urdf_path = assets_dir + "sim_data/urdf/test_robot/test_robot.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    // Global transform: scale and position the robot
    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.4;  // scale
    global_transform(1, 3) = 1.0;  // lift up
    urdf_importer.set_global_transform(global_transform);

    // Set root link as Fixed
    urdf_importer.set_root_fixed(true);

    // Enable revolute joints as Motor type
    urdf_importer.set_revolute_as_motor(true);
    urdf_importer.set_default_motor_speed(3.14);    // 0.5 rev/s
    urdf_importer.set_default_motor_strength(10.0);

    // Map each URDF link to a .msh tetrahedral mesh
    // All three links use the same cube.msh with different Young's moduli
    std::string cube_msh = assets_dir + "tetMesh/cube.msh";
    urdf_importer.set_mesh_override("base_link", {cube_msh, 1e7});
    urdf_importer.set_mesh_override("link1", {cube_msh, 1e6});
    urdf_importer.set_mesh_override("link2", {cube_msh, 1e6});

    // Import the scene - all links become ABD bodies
    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case7] URDF import failed!" << std::endl;
        std::abort();
    }

    // Print loaded info
    std::cout << "[set_case7] URDF test scene loaded successfully." << std::endl;
    std::cout << "[set_case7] Links loaded:" << std::endl;
    for(auto& [name, info] : urdf_importer.link_infos())
    {
        if(info.body_id >= 0)
        {
            std::cout << "  - " << name << " (body_id=" << info.body_id << ")" << std::endl;
        }
    }
    std::cout << "[set_case7] Joints:" << std::endl;
    for(auto& [name, info] : urdf_importer.joint_infos())
    {
        std::cout << "  - " << name << ": " << info.parent_link_name
                  << " -> " << info.child_link_name
                  << " (type=" << static_cast<int>(info.type)
                  << ", global_axis=[" << info.global_axis.x()
                  << "," << info.global_axis.y()
                  << "," << info.global_axis.z() << "])"
                  << std::endl;
    }
    xRot = 10.0f; yRot = -30.f; yTrans = -1.0f; zTrans = 0.5f;
}

// ==========================================================================
// Case 8: XArm6 URDF test
// Loads the xarm6 robot URDF. Uses cube.msh as placeholder for all links.
// Demonstrates: fixed joint (world->base), revolute joints (joint1-6).
// ==========================================================================
void set_case8_xarm6_test()
{
    gipc::UrdfSceneImporter urdf_importer;

    // Path to xarm6 URDF
    //std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm6_robot.urdf";
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    // Global transform: scale down and lift up
    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.3;  // scale
    global_transform(1, 3) = 1.5;  // lift up
    urdf_importer.set_global_transform(global_transform);

    // Root link (link_base, after world->link_base fixed joint) should be fixed
    urdf_importer.set_root_fixed(true);
    // Default non-root non-revolute links are Free
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);

    // Disable motor for now -- just test joint constraints under gravity
    urdf_importer.set_revolute_as_motor(false);
    //urdf_importer.set_default_motor_speed(1.57);    // ~0.25 rev/s
    //urdf_importer.set_default_motor_strength(10.0);

    // Default Young's modulus for all links loaded from .obj collision meshes
    urdf_importer.set_default_young_modulus(1e7);

    // No mesh overrides needed -- the importer will auto-detect .obj collision
    // meshes from the URDF and load them via the fan-tet approach.
    // To override a specific link with a tet mesh instead:
    //   urdf_importer.set_mesh_override("link_base", {"path/to/base.msh", 1e8});

    // Import the scene
    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case8] XArm6 URDF import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio = 1000.0;

    // Print summary
    std::cout << "[set_case8] XArm6 loaded successfully." << std::endl;
    std::cout << "[set_case8] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case8] Links:" << std::endl;
    for(auto& [name, info] : urdf_importer.link_infos())
    {
        if(info.body_id >= 0)
        {
            auto& mi = tetMesh.body_motor_infos[info.body_id];
            auto  bt = tetMesh.body_id_to_is_fixed[info.body_id];
            std::cout << "  - " << name << " (body=" << info.body_id
                      << ", boundary=" << static_cast<int>(bt)
                      << ", motor_axis=[" << mi.axis_x << "," << mi.axis_y << "," << mi.axis_z << "]"
                      << ", speed=" << mi.speed << ")" << std::endl;
        }
    }
    std::cout << "[set_case8] Joints:" << std::endl;
    for(auto& [name, info] : urdf_importer.joint_infos())
    {
        std::cout << "  - " << name << ": " << info.parent_link_name
                  << " -> " << info.child_link_name
                  << " (type=" << static_cast<int>(info.type)
                  << ", global_axis=[" << info.global_axis.x()
                  << "," << info.global_axis.y()
                  << "," << info.global_axis.z() << "])"
                  << std::endl;
    }
    xRot = 10.0f; yRot = -30.f; yTrans = -1.5f; zTrans = 0.5f;
}

// ==========================================================================
// Case 9: XArm7 with Gripper URDF test
// Loads the xarm7_with_gripper.urdf (7-DOF arm + parallel gripper).
// Demonstrates: .stl collision mesh loading, empty link pass-through.
// ==========================================================================
void set_case9_xarm7_gripper_test()
{
    gipc::UrdfSceneImporter urdf_importer;

    // Path to xarm7 with gripper URDF
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    // Global transform: scale down and lift up
    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.3;  // scale
    global_transform(1, 3) = 1.5;  // lift up
    urdf_importer.set_global_transform(global_transform);

    // Root link (link_base) should be fixed in world
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);

    // No motor for now -- just test joint constraints under gravity
    urdf_importer.set_revolute_as_motor(false);

    // Default Young's modulus for all links
    urdf_importer.set_default_young_modulus(1e7);

    // Import the scene
    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case9] XArm7+Gripper URDF import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio = 1000.0;

    // Print summary
    std::cout << "[set_case9] XArm7+Gripper loaded successfully." << std::endl;
    std::cout << "[set_case9] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case9] Links:" << std::endl;
    for(auto& [name, info] : urdf_importer.link_infos())
    {
        if(info.body_id >= 0)
        {
            auto  bt = tetMesh.body_id_to_is_fixed[info.body_id];
            std::cout << "  - " << name << " (body=" << info.body_id
                      << ", boundary=" << static_cast<int>(bt) << ")" << std::endl;
        }
        else
        {
            std::cout << "  - " << name << " (no geometry, skipped)" << std::endl;
        }
    }
    std::cout << "[set_case9] Joints:" << std::endl;
    for(auto& [name, info] : urdf_importer.joint_infos())
    {
        std::cout << "  - " << name << ": " << info.parent_link_name
                  << " -> " << info.child_link_name
                  << " (type=" << static_cast<int>(info.type) << ")" << std::endl;
    }
    std::cout << "[set_case9] Joint constraints: " << tetMesh.joint_constraints.size() << std::endl;
    xRot = 10.0f; yRot = -30.f; yTrans = -1.5f; zTrans = 0.5f;
}

// ==========================================================================
// Helper: load OBJ vertex positions (ignoring per-vertex colors if present)
// Optionally applies a 4x4 transform to each vertex.
// ==========================================================================
static std::vector<double3> load_obj_positions(const std::string&     obj_path,
                                               const Eigen::Matrix4d& transform = Eigen::Matrix4d::Identity())
{
    std::vector<double3> positions;
    std::ifstream        ifs(obj_path);
    if(!ifs.is_open())
    {
        std::cerr << "[load_obj_positions] Failed to open: " << obj_path << std::endl;
        return positions;
    }
    std::string line;
    while(std::getline(ifs, line))
    {
        if(line.size() >= 2 && line[0] == 'v' && line[1] == ' ')
        {
            double          x, y, z;
            std::istringstream ss(line.substr(2));
            ss >> x >> y >> z;  // ignore any extra color components
            Eigen::Vector4d p = transform * Eigen::Vector4d(x, y, z, 1.0);
            positions.push_back(make_double3(p(0), p(1), p(2)));
        }
    }
    std::cout << "[load_obj_positions] Loaded " << positions.size()
              << " vertices from " << obj_path << std::endl;
    return positions;
}

// ==========================================================================
// Helper: minimum distance from a point to any vertex in a point cloud
// ==========================================================================
static double min_dist_to_points(double3                      pt,
                                 const std::vector<double3>& verts)
{
    double best = 1e20;
    for(const auto& v : verts)
    {
        double dx = pt.x - v.x;
        double dy = pt.y - v.y;
        double dz = pt.z - v.z;
        double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
        if(d < best)
            best = d;
    }
    return best;
}

// ==========================================================================
// apply_per_tet_youngth_modulus  (GIPC equivalent of UIPC's "apply_to")
//
// Classify each tetrahedron in [tet_offset, tet_offset + tet_count) by
// centroid proximity to two reference point clouds ("hard" vs "soft"),
// then assign the corresponding Young's modulus.
// ==========================================================================
static void apply_per_tet_youngth_modulus(
    tetrahedra_obj&              mesh,
    int                          tet_offset,
    int                          tet_count,
    const std::vector<double3>&  hard_ref_verts,
    const std::vector<double3>&  soft_ref_verts,
    double                       hard_youngs,
    double                       soft_youngs)
{
    int hard_count = 0, soft_count = 0;
    for(int t = 0; t < tet_count; ++t)
    {
        int   gi  = tet_offset + t;
        auto& tet = mesh.tetrahedras[gi];

        // Compute tet centroid
        double3 centroid;
        centroid.x = (mesh.vertexes[tet.x].x + mesh.vertexes[tet.y].x
                      + mesh.vertexes[tet.z].x + mesh.vertexes[tet.w].x)
                     * 0.25;
        centroid.y = (mesh.vertexes[tet.x].y + mesh.vertexes[tet.y].y
                      + mesh.vertexes[tet.z].y + mesh.vertexes[tet.w].y)
                     * 0.25;
        centroid.z = (mesh.vertexes[tet.x].z + mesh.vertexes[tet.y].z
                      + mesh.vertexes[tet.z].z + mesh.vertexes[tet.w].z)
                     * 0.25;

        double d_hard = min_dist_to_points(centroid, hard_ref_verts);
        double d_soft = min_dist_to_points(centroid, soft_ref_verts);

        if(d_hard <= d_soft)
        {
            mesh.vert_youngth_modules[gi] = hard_youngs;
            ++hard_count;
        }
        else
        {
            mesh.vert_youngth_modules[gi] = soft_youngs;
            ++soft_count;
        }
    }
    std::cout << "[apply_per_tet_youngth_modulus] "
              << hard_count << " hard tets, "
              << soft_count << " soft tets" << std::endl;
}

// ==========================================================================
// Case 10: UIPC-style Soft Gripper Demo
//
// Replicates the UIPC hello_vbd soft gripper scene in GIPC.
// Two fingers (ABD rigid base + FEM heterogeneous-stiffness combined mesh)
// grip a cup (ABD). Per-tet Young's modulus is assigned based on proximity
// to part2 (hard) and part3 (soft) reference surfaces.
// ==========================================================================
void set_case10_uipc_demo()
{
    // ==========================================================
    // UIPC-style soft gripper demo (ABD rigid base + FEM soft body):
    //   ABD cube (top, Animated boundary -- driven upward by soft
    //             translation penalty, analogous to UIPC SoftTransformConstraint)
    //   FEM cube (bottom, soft E=1e5, connected to ABD via stitch springs)
    //
    // ABD must be loaded before FEM.
    // ==========================================================

    ipc.pcg_data.P_type = 0;  // bypass metis_sort

    std::string tetmesh_dir = assets_dir + "tetMesh/";
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    std::string cube_msh = tetmesh_dir + "cube.msh";
    // cube.msh actual vertex bounds:
    //   x: [-0.2, 0.2], y: [0.1, 0.5], z: [-0.2, 0.2]
    //   top face (y=0.5): verts 1-4, bottom face (y=0.1): verts 5-8
    double mesh_y_min = 0.1;  // bottom face y in original mesh
    double mesh_y_max = 0.5;  // top face y in original mesh

    double scale       = 0.3;   // scaling factor
    double gap         = 0.3;   // visible gap between cubes (applied after stitch)
    double soft_E      = 1e5;   // FEM cube: soft
    double lift_speed  = 0.1;   // m/s -- ABD upward velocity
    double anim_strength = 1e6; // Animated penalty stiffness

    // ----------------------------------------------------------
    // 1) Load ABD cube (top, rigid)
    //    Position so bottom face (y_min) lands at y=0.
    //    After transform: y_bottom = scale*mesh_y_min + ty = 0 → ty = -scale*mesh_y_min
    // ----------------------------------------------------------
    int abdA_vert_start = tetMesh.vertexNum;
    int abdA_body_id = -1;
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, -scale * mesh_y_min, 0});
        t.scale(scale);
        abdA_body_id = static_cast<int>(tetMesh.body_id_to_is_fixed.size());
        importer.load_geometry(tetMesh, 3, gipc::BodyType::ABD, t.matrix(),
                               1e8 /* YoungthM (unused for ABD) */, cube_msh,
                               ipc.pcg_data.P_type, BodyBoundaryType::Animated);
    }
    int abdA_vert_end = tetMesh.vertexNum;
    std::cout << "[case10] ABD cube (Animated) loaded, body_id=" << abdA_body_id
              << ", verts [" << abdA_vert_start << ", " << abdA_vert_end << ")"
              << std::endl;

    // Set Animated motor params: [vx, vy, vz, strength, 0]
    if(abdA_body_id >= 0 && abdA_body_id < static_cast<int>(tetMesh.body_motor_infos.size()))
    {
        auto& mi = tetMesh.body_motor_infos[abdA_body_id];
        mi.axis_x   = 0.0;          // vx
        mi.axis_y   = lift_speed;    // vy
        mi.axis_z   = 0.0;          // vz
        mi.speed    = anim_strength; // stored in params[3] → anim_strength
        mi.strength = 0.0;          // params[4] unused for Animated
    }

    // ----------------------------------------------------------
    // 2) Load FEM cube (bottom, soft)
    //    Position so top face (y_max) lands at y=0.
    //    After transform: y_top = scale*mesh_y_max + ty = 0 → ty = -scale*mesh_y_max
    // ----------------------------------------------------------
    int femB_vert_start = tetMesh.vertexNum;
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, -scale * mesh_y_max, 0});
        t.scale(scale);
        importer.load_geometry(tetMesh, 3, gipc::BodyType::FEM, t.matrix(),
                               soft_E, cube_msh, ipc.pcg_data.P_type);
    }
    int femB_vert_end = tetMesh.vertexNum;
    std::cout << "[case10] FEM cube (soft) loaded, verts ["
              << femB_vert_start << ", " << femB_vert_end << ")" << std::endl;

    // Enable gravity on FEM cube only (ABD has its own gravity via q_tilde)
    for(int i = 0; i < tetMesh.vertexNum; i++)
        tetMesh.apply_gravity[i] = 0;

    // Debug: print all vertex positions
    std::cout << "[case10] ABD vertices:" << std::endl;
    for(int v = abdA_vert_start; v < abdA_vert_end; ++v)
    {
        auto& p = tetMesh.vertexes[v];
        std::cout << "  v" << v << ": (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }
    std::cout << "[case10] FEM vertices:" << std::endl;
    for(int v = femB_vert_start; v < femB_vert_end; ++v)
    {
        auto& p = tetMesh.vertexes[v];
        std::cout << "  v" << v << ": (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }

    // ----------------------------------------------------------
    // 3) Find stitch pairs (mutual NN on the facing surfaces)
    //    Match by x/z distance only (ignoring y gap between cubes)
    //    ABD bottom face ↔ FEM top face
    // ----------------------------------------------------------
    std::vector<std::pair<int, int>> stitch_pairs;  // (femB_idx, abdA_idx)
    {
        double threshold = scale * 0.15;  // x/z matching threshold
        int A_count = abdA_vert_end - abdA_vert_start;
        int B_count = femB_vert_end - femB_vert_start;

        for(int b = 0; b < B_count; ++b)
        {
            auto& pb = tetMesh.vertexes[femB_vert_start + b];
            double best_d = 1e20;
            int    best_a = -1;
            for(int a = 0; a < A_count; ++a)
            {
                auto& pa = tetMesh.vertexes[abdA_vert_start + a];
                double dx = pb.x - pa.x, dz = pb.z - pa.z;
                double d  = std::sqrt(dx * dx + dz * dz);  // x/z only
                if(d < best_d) { best_d = d; best_a = a; }
            }
            if(best_d < threshold && best_a >= 0)
            {
                // Reverse check: ensure mutual nearest neighbor
                auto& pa = tetMesh.vertexes[abdA_vert_start + best_a];
                double rev_best = 1e20;
                int    rev_b    = -1;
                for(int b2 = 0; b2 < B_count; ++b2)
                {
                    auto& pb2 = tetMesh.vertexes[femB_vert_start + b2];
                    double dx = pa.x - pb2.x, dz = pa.z - pb2.z;
                    double d  = std::sqrt(dx * dx + dz * dz);  // x/z only
                    if(d < rev_best) { rev_best = d; rev_b = b2; }
                }
                if(rev_b == b)
                {
                    stitch_pairs.emplace_back(femB_vert_start + b,
                                              abdA_vert_start + best_a);
                    break;
                    std::cout << "[case10] Stitch: FEM v" << (femB_vert_start + b)
                              << " <-> ABD v" << (abdA_vert_start + best_a)
                              << "  dist=" << best_d << std::endl;
                }
            }
        }
    }
    std::cout << "[case10] Total stitch pairs: " << stitch_pairs.size() << std::endl;

    // ----------------------------------------------------------
    // 4) Shift ABD cube upward by 'gap' (after stitch pair finding)
    // ----------------------------------------------------------
    if(gap > 0)
    {
        for(int v = abdA_vert_start; v < abdA_vert_end; ++v)
            tetMesh.vertexes[v].y += gap;
        std::cout << "[case10] ABD cube shifted up by " << gap << std::endl;
    }

    // ----------------------------------------------------------
    // 5) Build soft constraints (stitch springs only)
    //    Each stitch spring: FEM vertex tracks ABD vertex + rest offset
    //    The soft constraint target updates each step via
    //    update_soft_constraint_target_position using stitch_paired_vertex.
    // ----------------------------------------------------------
    std::vector<int>     stitch_paired_vertex_map;
    std::vector<double3> stitch_rest_offsets;
    std::vector<int>     stitch_abd_body_ids;

    for(auto& [fem_idx, abd_idx] : stitch_pairs)
    {
        auto& fp = tetMesh.vertexes[fem_idx];
        auto& ap = tetMesh.vertexes[abd_idx];  // already shifted up by gap
        double3 rest_off = make_double3(fp.x - ap.x, fp.y - ap.y, fp.z - ap.z);

        tetMesh.targetIndex.push_back(fem_idx);
        tetMesh.targetPos.push_back(fp);  // initial target = current pos (zero force)
        stitch_paired_vertex_map.push_back(abd_idx);
        stitch_rest_offsets.push_back(rest_off);
        stitch_abd_body_ids.push_back(abdA_body_id);

        std::cout << "[case10] Rest offset: (" << rest_off.x << ", "
                  << rest_off.y << ", " << rest_off.z << ")" << std::endl;
    }

    tetMesh.softNum = tetMesh.targetIndex.size();
    d_tetMesh.stitch_paired_vertex = stitch_paired_vertex_map;
    d_tetMesh.stitch_rest_offset   = stitch_rest_offsets;
    d_tetMesh.stitch_abd_body_id   = stitch_abd_body_ids;

    std::cout << "[case10] Soft constraints (stitch): " << tetMesh.softNum << std::endl;

    // Stitch spring stiffness
    ipc.softMotionRate = 1e6;

    // Simulation parameters
    ipc.relative_dhat = 1e-3;
    ipc.PoissonRate   = 0.45;

    std::cout << "[case10] ABD(Animated) + FEM stitch demo setup complete."
              << std::endl;
    xRot = 15.0f; yRot = -30.f; yTrans = 0.0f; zTrans = 1.5f;
}

// ==========================================================================
// Case 11: Full Soft Gripper Scene
// ==========================================================================
void set_case11_gripper()
{
    ipc.pcg_data.P_type = 0;  // bypass metis_sort

    std::string sim_tetmesh = assets_dir + "sim_data/tetmesh/";
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    // --- Mesh paths ---------------------------------------------
    std::string part1_msh     = sim_tetmesh + "softgriper_part1.msh";
    std::string combined_msh  = sim_tetmesh + "softgriper_part2_blobal.msh";
    std::string cup_msh       = sim_tetmesh + "softgriper_cup.msh";
    std::string part2_obj     = sim_tetmesh + "softgriper_part2.obj";
    std::string part3_obj     = sim_tetmesh + "softgriper_part3.obj";

    // --- Scene parameters ---------------------------------------
    double finger_dist    = 0.3;
    double finger_y       = -0.95;   // base center height (ground at y=-1)
    double finger_z_off   = -0.05;
    double part1_y_off    = -0.093;
    double combined_y_off = -0.0933;
    double combined_scale = 0.97;

    double separation_dist = 0.02; // gap between ABD base and FEM body (meters)
    bool   enable_stitch    = true;  // stitch springs between ABD base and FEM body
    bool   enable_animation  = true;  // ABD animated movement

    double hard_E = 1e7;
    double soft_E = 1e6;
    double base_E = 1e8;
    double cup_E  = 1e8;

    double anim_speed    = 1.0;
    double anim_strength = 1e5;
    int    switch_frame  = 27;  // frame 0-49: grip inward, frame 50+: lift upward

    // --- Helper lambdas -----------------------------------------
    auto compute_bbox_center = [&](int v_start, int v_end) -> Eigen::Vector3d {
        Eigen::Vector3d mn(1e20, 1e20, 1e20), mx(-1e20, -1e20, -1e20);
        for (int v = v_start; v < v_end; ++v) {
            auto& p = tetMesh.vertexes[v];
            mn.x() = std::min(mn.x(), p.x); mn.y() = std::min(mn.y(), p.y); mn.z() = std::min(mn.z(), p.z);
            mx.x() = std::max(mx.x(), p.x); mx.y() = std::max(mx.y(), p.y); mx.z() = std::max(mx.z(), p.z);
        }
        return (mn + mx) * 0.5;
    };
    auto rotate_verts_around = [&](int v_start, int v_end, const Eigen::Matrix3d& R, const Eigen::Vector3d& center) {
        for (int v = v_start; v < v_end; ++v) {
            auto& p = tetMesh.vertexes[v];
            Eigen::Vector3d pos(p.x, p.y, p.z);
            pos = R * (pos - center) + center;
            p = make_double3(pos.x(), pos.y(), pos.z());
        }
    };
    auto translate_verts = [&](int v_start, int v_end, const Eigen::Vector3d& t) {
        for (int v = v_start; v < v_end; ++v) {
            tetMesh.vertexes[v].x += t.x();
            tetMesh.vertexes[v].y += t.y();
            tetMesh.vertexes[v].z += t.z();
        }
    };

    // =============================================================
    // PHASE A: Load all ABD bodies first (GIPC requirement)
    // =============================================================

    // 1. Cup (ABD, Free)
    int cup_vert_start = tetMesh.vertexNum;
    int cup_body_id    = static_cast<int>(tetMesh.body_id_to_is_fixed.size());
    {
        Transform t = Transform::Identity();
        importer.load_geometry(tetMesh, 3, gipc::BodyType::ABD, t.matrix(),
                               cup_E, cup_msh, ipc.pcg_data.P_type, BodyBoundaryType::Free);
    }
    int cup_vert_end = tetMesh.vertexNum;
    {
        Eigen::Vector3d mn(1e20, 1e20, 1e20), mx(-1e20, -1e20, -1e20);
        for (int v = cup_vert_start; v < cup_vert_end; ++v) {
            auto& p = tetMesh.vertexes[v];
            mn.x() = std::min(mn.x(), p.x); mn.y() = std::min(mn.y(), p.y); mn.z() = std::min(mn.z(), p.z);
            mx.x() = std::max(mx.x(), p.x); mx.y() = std::max(mx.y(), p.y); mx.z() = std::max(mx.z(), p.z);
        }
        Eigen::Vector3d shift(-(mn.x() + mx.x()) * 0.5, -mn.y() - 0.995, -(mn.z() + mx.z()) * 0.5);
        translate_verts(cup_vert_start, cup_vert_end, shift);
    }
    std::cout << "[case11] Cup: body=" << cup_body_id << ", verts [" << cup_vert_start << "," << cup_vert_end << ")" << std::endl;

    // 2. Finger 0 base (ABD)
    BodyBoundaryType base_btype = enable_animation ? BodyBoundaryType::Animated : BodyBoundaryType::Free;
    int f0_base_vs = tetMesh.vertexNum;
    int f0_base_id = static_cast<int>(tetMesh.body_id_to_is_fixed.size());
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, part1_y_off, 0});
        importer.load_geometry(tetMesh, 3, gipc::BodyType::ABD, t.matrix(),
                               base_E, part1_msh, ipc.pcg_data.P_type, base_btype);
    }
    int f0_base_ve = tetMesh.vertexNum;

    // 3. Finger 1 base (ABD)
    int f1_base_vs = tetMesh.vertexNum;
    int f1_base_id = static_cast<int>(tetMesh.body_id_to_is_fixed.size());
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, part1_y_off, 0});
        importer.load_geometry(tetMesh, 3, gipc::BodyType::ABD, t.matrix(),
                               base_E, part1_msh, ipc.pcg_data.P_type, base_btype);
    }
    int f1_base_ve = tetMesh.vertexNum;

    std::cout << "[case11] F0 base: body=" << f0_base_id << ", F1 base: body=" << f1_base_id << std::endl;

    // =============================================================
    // PHASE B: Load FEM bodies
    // =============================================================

    // 4. Finger 0 FEM
    int f0_fem_vs = tetMesh.vertexNum;
    int f0_fem_ts = tetMesh.tetrahedraNum;
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, combined_y_off, 0});
        t.scale(combined_scale);
        importer.load_geometry(tetMesh, 3, gipc::BodyType::FEM, t.matrix(),
                               hard_E, combined_msh, ipc.pcg_data.P_type);
    }
    int f0_fem_ve = tetMesh.vertexNum;
    int f0_fem_te = tetMesh.tetrahedraNum;

    // 5. Finger 1 FEM
    int f1_fem_vs = tetMesh.vertexNum;
    int f1_fem_ts = tetMesh.tetrahedraNum;
    {
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0, combined_y_off, 0});
        t.scale(combined_scale);
        importer.load_geometry(tetMesh, 3, gipc::BodyType::FEM, t.matrix(),
                               hard_E, combined_msh, ipc.pcg_data.P_type);
    }
    int f1_fem_ve = tetMesh.vertexNum;
    int f1_fem_te = tetMesh.tetrahedraNum;

    std::cout << "[case11] F0 FEM: verts[" << f0_fem_vs << "," << f0_fem_ve
              << ") tets[" << f0_fem_ts << "," << f0_fem_te << ")" << std::endl;
    std::cout << "[case11] F1 FEM: verts[" << f1_fem_vs << "," << f1_fem_ve
              << ") tets[" << f1_fem_ts << "," << f1_fem_te << ")" << std::endl;

    // =============================================================
    // PHASE C: Per-tet heterogeneous Young's modulus
    // (BEFORE rotation -- ref verts & mesh share the same local frame)
    // =============================================================
    {
        Eigen::Matrix4d ref_t = Eigen::Matrix4d::Identity();
        ref_t.block<3, 3>(0, 0) *= combined_scale;
        ref_t(1, 3) = combined_y_off;

        auto hard_ref = load_obj_positions(part2_obj, ref_t);
        auto soft_ref = load_obj_positions(part3_obj, ref_t);

        std::cout << "[case11] Ref verts: hard=" << hard_ref.size() << " soft=" << soft_ref.size() << std::endl;

        apply_per_tet_youngth_modulus(tetMesh, f0_fem_ts, f0_fem_te - f0_fem_ts,
                                     hard_ref, soft_ref, hard_E, soft_E);
        apply_per_tet_youngth_modulus(tetMesh, f1_fem_ts, f1_fem_te - f1_fem_ts,
                                     hard_ref, soft_ref, hard_E, soft_E);
    }

    // =============================================================
    // PHASE D: Stitch springs (before rotation -- find pairs in local frame)
    // =============================================================
    struct StitchInfo { std::vector<std::pair<int, int>> pairs; };

    auto find_stitch_pairs = [&](int abd_vs, int abd_ve, int fem_vs, int fem_ve, double thresh) -> StitchInfo {
        StitchInfo si;
        for (int b = fem_vs; b < fem_ve; ++b) {
            auto& pb = tetMesh.vertexes[b];
            double best_d = 1e20; int best_a = -1;
            for (int a = abd_vs; a < abd_ve; ++a) {
                auto& pa = tetMesh.vertexes[a];
                double dx = pb.x - pa.x, dy = pb.y - pa.y, dz = pb.z - pa.z;
                double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < best_d) { best_d = d; best_a = a; }
            }
            if (best_d < thresh && best_a >= 0) {
                // Reverse check: is b the closest FEM vert to best_a?
                auto& pa = tetMesh.vertexes[best_a];
                double rev_best = 1e20; int rev_b = -1;
                for (int b2 = fem_vs; b2 < fem_ve; ++b2) {
                    auto& pb2 = tetMesh.vertexes[b2];
                    double dx = pa.x - pb2.x, dy = pa.y - pb2.y, dz = pa.z - pb2.z;
                    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d < rev_best) { rev_best = d; rev_b = b2; }
                }
                if(rev_b == b)
                {
                    si.pairs.emplace_back(b, best_a);
                    //break;
                }
            }
        }
        return si;
    };

    double stitch_thresh = 0.005;
    auto f0_stitch = find_stitch_pairs(f0_base_vs, f0_base_ve, f0_fem_vs, f0_fem_ve, stitch_thresh);
    auto f1_stitch = find_stitch_pairs(f1_base_vs, f1_base_ve, f1_fem_vs, f1_fem_ve, stitch_thresh);
    std::cout << "[case11] Stitch pairs: F0=" << f0_stitch.pairs.size()
              << " F1=" << f1_stitch.pairs.size() << std::endl;
    // Print each pair's vertex indices and positions for debugging
    auto print_pairs = [&](const char* name, const StitchInfo& si) {
        for (size_t i = 0; i < si.pairs.size(); ++i) {
            auto [fem_idx, abd_idx] = si.pairs[i];
            auto& fp = tetMesh.vertexes[fem_idx];
            auto& ap = tetMesh.vertexes[abd_idx];
            double dx = fp.x - ap.x, dy = fp.y - ap.y, dz = fp.z - ap.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            std::cout << "  " << name << " pair[" << i << "]: FEM v" << fem_idx
                      << " (" << fp.x << "," << fp.y << "," << fp.z << ") <-> ABD v"
                      << abd_idx << " (" << ap.x << "," << ap.y << "," << ap.z
                      << ")  dist=" << dist << std::endl;
        }
    };
    print_pairs("F0", f0_stitch);
    print_pairs("F1", f1_stitch);

    // =============================================================
    // PHASE D.5: Separate FEM from ABD base to avoid intersection
    // In local frame the finger is vertical (Y-up), base is at top,
    // FEM hangs below. Translate FEM further down by separation_dist.
    // =============================================================
    translate_verts(f0_fem_vs, f0_fem_ve, Eigen::Vector3d(0, -separation_dist, 0));
    translate_verts(f1_fem_vs, f1_fem_ve, Eigen::Vector3d(0, -separation_dist, 0));
    std::cout << "[case11] Separated FEM from ABD base by " << separation_dist << "m" << std::endl;

    // =============================================================
    // PHASE E: Rotations & Translations
    // =============================================================
    Eigen::Matrix3d rot_x_neg90;
    {
        double a = -3.14159265358979323846 * 0.5;
        rot_x_neg90 << 1, 0, 0,
                        0, std::cos(a), -std::sin(a),
                        0, std::sin(a),  std::cos(a);
    }
    Eigen::Matrix3d rot_z_180;
    rot_z_180 << -1, 0, 0,  0, -1, 0,  0, 0, 1;

    // Finger 0 (Right): -90 X, translate to (-finger_dist, finger_y, finger_z_off)
    {
        // Rotate around combined bbox center
        Eigen::Vector3d center = compute_bbox_center(f0_base_vs, f0_fem_ve);
        rotate_verts_around(f0_base_vs, f0_base_ve, rot_x_neg90, center);
        rotate_verts_around(f0_fem_vs,  f0_fem_ve,  rot_x_neg90, center);
        // Align by ABD base center (not combined center) so both fingers match
        Eigen::Vector3d base_center = compute_bbox_center(f0_base_vs, f0_base_ve);
        Eigen::Vector3d shift = Eigen::Vector3d(-finger_dist, finger_y, finger_z_off) - base_center;
        translate_verts(f0_base_vs, f0_base_ve, shift);
        translate_verts(f0_fem_vs,  f0_fem_ve,  shift);
    }

    // Finger 1 (Left): 180Z * -90X, translate to (+finger_dist, finger_y, finger_z_off)
    {
        Eigen::Matrix3d R1 = rot_z_180 * rot_x_neg90;
        Eigen::Vector3d center = compute_bbox_center(f1_base_vs, f1_fem_ve);
        rotate_verts_around(f1_base_vs, f1_base_ve, R1, center);
        rotate_verts_around(f1_fem_vs,  f1_fem_ve,  R1, center);
        // Align by ABD base center
        Eigen::Vector3d base_center = compute_bbox_center(f1_base_vs, f1_base_ve);
        Eigen::Vector3d shift = Eigen::Vector3d(finger_dist, finger_y, finger_z_off) - base_center;
        translate_verts(f1_base_vs, f1_base_ve, shift);
        translate_verts(f1_fem_vs,  f1_fem_ve,  shift);
    }

    // Print bounding boxes for all objects to verify no ground intersection
    auto print_bbox = [&](const char* name, int vs, int ve) {
        Eigen::Vector3d mn(1e20,1e20,1e20), mx(-1e20,-1e20,-1e20);
        for (int v = vs; v < ve; ++v) {
            auto& p = tetMesh.vertexes[v];
            mn.x()=std::min(mn.x(),p.x); mn.y()=std::min(mn.y(),p.y); mn.z()=std::min(mn.z(),p.z);
            mx.x()=std::max(mx.x(),p.x); mx.y()=std::max(mx.y(),p.y); mx.z()=std::max(mx.z(),p.z);
        }
        std::cout << "[case11] " << name << " bbox: ("
                  << mn.x() << "," << mn.y() << "," << mn.z() << ") -> ("
                  << mx.x() << "," << mx.y() << "," << mx.z() << ")" << std::endl;
    };
    print_bbox("Cup",      cup_vert_start, cup_vert_end);
    print_bbox("F0 base",  f0_base_vs, f0_base_ve);
    print_bbox("F0 FEM",   f0_fem_vs,  f0_fem_ve);
    print_bbox("F1 base",  f1_base_vs, f1_base_ve);
    print_bbox("F1 FEM",   f1_fem_vs,  f1_fem_ve);

    // =============================================================
    // PHASE F: Build soft constraints (stitch springs)
    // =============================================================
    if (enable_stitch) {
        std::vector<int>     stitch_paired_vertex_map;
        std::vector<double3> stitch_rest_offsets;
        std::vector<int>     stitch_abd_body_ids;

        auto add_stitch = [&](const StitchInfo& si, int abd_body_id) {
            for (auto& [fem_idx, abd_idx] : si.pairs) {
                auto& fp = tetMesh.vertexes[fem_idx];
                auto& ap = tetMesh.vertexes[abd_idx];
                double3 rest_off = make_double3(fp.x - ap.x, fp.y - ap.y, fp.z - ap.z);
                tetMesh.targetIndex.push_back(fem_idx);
                tetMesh.targetPos.push_back(fp);
                stitch_paired_vertex_map.push_back(abd_idx);
                stitch_rest_offsets.push_back(rest_off);
                stitch_abd_body_ids.push_back(abd_body_id);
            }
        };
        add_stitch(f0_stitch, f0_base_id);
        add_stitch(f1_stitch, f1_base_id);

        tetMesh.softNum = static_cast<int>(tetMesh.targetIndex.size());
        d_tetMesh.stitch_paired_vertex = stitch_paired_vertex_map;
        d_tetMesh.stitch_rest_offset   = stitch_rest_offsets;
        d_tetMesh.stitch_abd_body_id   = stitch_abd_body_ids;
        // UIPC effective stiffness = kappa * dt^2 = 1e8 * 0.01^2 = 1e4
        // GIPC effective stiffness = softMotionRate * rate^2 = softMotionRate * 1.0
        // So set softMotionRate = 1e4 to match UIPC
        ipc.softMotionRate = 1e4;

        std::cout << "[case11] Total stitch soft constraints: " << tetMesh.softNum << std::endl;
    } else {
        tetMesh.softNum = 0;
        std::cout << "[case11] Stitch springs DISABLED for speed test" << std::endl;
    }

    // =============================================================
    // PHASE G: Animation (Animated ABD motor params + per-frame callback)
    //
    // Motor params now store ABSOLUTE target position [tx, ty, tz, strength, 0].
    // The pre_step_functor accumulates displacement from the initial centroid
    // each step, computing: target = initial_pos + sum(velocity * dt).
    // =============================================================
    d_tetMesh.m_body_count = static_cast<int>(tetMesh.body_id_to_is_fixed.size());

    if (enable_animation) {
        // Compute initial centroid of each ABD base (average of vertices)
        auto compute_centroid = [&](int vs, int ve) -> Eigen::Vector3d {
            Eigen::Vector3d c = Eigen::Vector3d::Zero();
            for (int v = vs; v < ve; ++v) {
                auto& p = tetMesh.vertexes[v];
                c += Eigen::Vector3d(p.x, p.y, p.z);
            }
            return c / (ve - vs);
        };
        Eigen::Vector3d f0_init_pos = compute_centroid(f0_base_vs, f0_base_ve);
        Eigen::Vector3d f1_init_pos = compute_centroid(f1_base_vs, f1_base_ve);
        std::cout << "[case11] F0 base initial centroid: ("
                  << f0_init_pos.x() << "," << f0_init_pos.y() << "," << f0_init_pos.z() << ")" << std::endl;
        std::cout << "[case11] F1 base initial centroid: ("
                  << f1_init_pos.x() << "," << f1_init_pos.y() << "," << f1_init_pos.z() << ")" << std::endl;

        // Initial motor params: target = initial position (no displacement yet)
        auto& mi0 = tetMesh.body_motor_infos[f0_base_id];
        mi0.axis_x = f0_init_pos.x(); mi0.axis_y = f0_init_pos.y(); mi0.axis_z = f0_init_pos.z();
        mi0.speed = anim_strength; mi0.strength = 0;

        auto& mi1 = tetMesh.body_motor_infos[f1_base_id];
        mi1.axis_x = f1_init_pos.x(); mi1.axis_y = f1_init_pos.y(); mi1.axis_z = f1_init_pos.z();
        mi1.speed = anim_strength; mi1.strength = 0;

        // Per-frame callback: accumulate displacement, compute absolute target
        d_tetMesh.pre_step_functor =
            [f0_id = f0_base_id, f1_id = f1_base_id,
             f0_pos0 = f0_init_pos, f1_pos0 = f1_init_pos,
             switch_frame, anim_speed, anim_strength, body_count = d_tetMesh.m_body_count,
             f0_disp = Eigen::Vector3d(0,0,0), f1_disp = Eigen::Vector3d(0,0,0)]
            (int step_id, double ipc_dt, double* body_motor_params_gpu, int bc) mutable
        {
            // Compute velocity for this step
            Eigen::Vector3d f0_vel, f1_vel;
            if (step_id < switch_frame) {
                // Grip phase: move inward (F0 +x, F1 -x)
                f0_vel = Eigen::Vector3d( anim_speed, 0, 0);
                f1_vel = Eigen::Vector3d(-anim_speed, 0, 0);
            } else {
                // Lift phase: move up (both +y), stop horizontal
                f0_vel = Eigen::Vector3d(0, anim_speed, 0);
                f1_vel = Eigen::Vector3d(0, anim_speed, 0);
            }

            // Accumulate displacement
            f0_disp += f0_vel * ipc_dt;
            f1_disp += f1_vel * ipc_dt;

            // Absolute target = initial position + accumulated displacement
            Eigen::Vector3d f0_target = f0_pos0 + f0_disp;
            Eigen::Vector3d f1_target = f1_pos0 + f1_disp;

            // Upload
            std::vector<double> params(body_count * 5, 0.0);
            params[f0_id * 5 + 0] = f0_target.x();
            params[f0_id * 5 + 1] = f0_target.y();
            params[f0_id * 5 + 2] = f0_target.z();
            params[f0_id * 5 + 3] = anim_strength;
            params[f1_id * 5 + 0] = f1_target.x();
            params[f1_id * 5 + 1] = f1_target.y();
            params[f1_id * 5 + 2] = f1_target.z();
            params[f1_id * 5 + 3] = anim_strength;
            CUDA_SAFE_CALL(cudaMemcpy(body_motor_params_gpu, params.data(),
                                      body_count * 5 * sizeof(double), cudaMemcpyHostToDevice));
            if (step_id <= 3 || step_id == switch_frame || step_id == switch_frame + 1)
                printf("[anim] step=%d  F0 target=(%.4f,%.4f,%.4f)  F1 target=(%.4f,%.4f,%.4f)\n",
                       step_id, f0_target.x(), f0_target.y(), f0_target.z(),
                       f1_target.x(), f1_target.y(), f1_target.z());
        };
        std::cout << "[case11] Animation ENABLED (absolute target mode)" << std::endl;
    } else {
        std::cout << "[case11] Animation DISABLED for speed test" << std::endl;
    }

    // =============================================================
    // PHASE H: Final simulation parameters
    // =============================================================
    for (int i = 0; i < tetMesh.vertexNum; i++)
        tetMesh.apply_gravity[i] = 1;
    // Note: With absolute target mode, gravity on ABD bases is fine --
    // the constraint pulls toward the absolute target regardless of gravity.

    ipc.relative_dhat = 1e-3;
    ipc.PoissonRate   = 0.45;
    // Match UIPC default friction coefficient (0.5)
    ipc.frictionRate    = 0.5;
    ipc.gd_frictionRate = 0.5;

    // =============================================================
    // PHASE I: Collision exclusion (DISABLED for debugging)
    //  Assign FEM body IDs so intersection printf shows which FEM body,
    //  but do NOT add collision exclusion pairs.
    //  Body layout:
    //    0 = Cup, 1 = F0 base, 2 = F1 base, 3 = F0 FEM, 4 = F1 FEM
    // =============================================================
    {
        int abd_body_count = static_cast<int>(tetMesh.abd_fem_count_info.abd_body_num);
        int f0_fem_body_id = abd_body_count;     // 3
        int f1_fem_body_id = abd_body_count + 1; // 4

        // Assign body IDs so intersection printf is useful
        for (int v = f0_fem_vs; v < f0_fem_ve; ++v)
            tetMesh.point_id_to_body_id[v] = f0_fem_body_id;
        for (int v = f1_fem_vs; v < f1_fem_ve; ++v)
            tetMesh.point_id_to_body_id[v] = f1_fem_body_id;

        std::cout << "[case11] FEM body IDs assigned: F0 FEM=" << f0_fem_body_id
                  << " (verts " << f0_fem_vs << "-" << f0_fem_ve << ")"
                  << " F1 FEM=" << f1_fem_body_id
                  << " (verts " << f1_fem_vs << "-" << f1_fem_ve << ")" << std::endl;
        std::cout << "[case11] Collision exclusion: NONE (debug mode)" << std::endl;
    }

    std::cout << "[case11] Soft gripper scene setup complete. Bodies=" << d_tetMesh.m_body_count
              << " Verts=" << tetMesh.vertexNum << " Tets=" << tetMesh.tetrahedraNum << std::endl;
    xRot = 10.0f; yRot = -30.f; yTrans = 0.5f; zTrans = 2.0f;
}

// ==========================================================================
// Case 12: Multi-Cube Revolute Chain Test (N cubes + UI angle sliders)
//
// A chain of N ABD cubes connected by revolute joints.
// - cube 0: Fixed (base)
// - cubes 1..N-1: Free, each connected to previous cube by a revolute joint
// Each joint has its own UI slider for angle control.
// Alternating joint axes (Z, X, Z, X, ...) to test 3D articulation.
// ==========================================================================
void set_case12_two_cube_revolute_test()
{
    ipc.pcg_data.P_type = 0;

    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    std::string cube_msh = assets_dir + "tetMesh/cube.msh";
    double      scale    = 0.25;

    // cube.msh local bounds: x,z: [-0.2, 0.2], y: [0.1, 0.5]
    // cube height = 0.4 * scale, we place cubes touching along Y.
    double cube_height = 0.4 * scale;  // 0.1 in world (scale=0.25)

    constexpr int NUM_CUBES = 5;
    std::vector<int> body_ids(NUM_CUBES);

    // Create cube chain along +Y axis.
    // cube 0: base at y = [-0.1, 0] (fixed)
    // cube i: stacked on top, y_bottom = cube_height * i
    for(int i = 0; i < NUM_CUBES; i++)
    {
        body_ids[i] = static_cast<int>(tetMesh.body_id_to_is_fixed.size());

        Transform t = Transform::Identity();
        double y_offset = cube_height * i - 0.5 * scale;
        t.translate(Eigen::Vector3d{0.0, y_offset, 0.0});
        t.scale(scale);
        importer.load_geometry(tetMesh,
                               3,
                               gipc::BodyType::ABD,
                               t.matrix(),
                               1e8,
                               cube_msh,
                               ipc.pcg_data.P_type,
                               (i == 0) ? BodyBoundaryType::Fixed
                                        : BodyBoundaryType::Free);
    }

    // Create revolute joints between consecutive cubes.
    // Joint i connects cube i (parent) to cube i+1 (child).
    // Joint anchor = top face of cube i = bottom face of cube i+1.
    for(int i = 0; i < NUM_CUBES - 1; i++)
    {
        double joint_y = cube_height * (i + 1) - 0.5 * scale + 0.1 * scale;

        // Alternate axes: even joints rotate around Z, odd around X
        Eigen::Vector3d axis_dir = (i % 2 == 0) ? Eigen::Vector3d::UnitZ()
                                                 : Eigen::Vector3d::UnitX();
        Eigen::Vector3d n_dir    = (i % 2 == 0) ? Eigen::Vector3d::UnitX()
                                                 : Eigen::Vector3d::UnitY();

        JointConstraintHostInfo jc;
        jc.parent_body_id = body_ids[i];
        jc.child_body_id  = body_ids[i + 1];
        jc.type           = JointConstraintHostInfo::Type::Revolute;
        jc.num_points     = 2;
        jc.world_anchor[0] = Eigen::Vector3d(0, joint_y, 0) - 0.5 * axis_dir;
        jc.world_anchor[1] = Eigen::Vector3d(0, joint_y, 0) + 0.5 * axis_dir;
        jc.point_weight[0] = 1.0;
        jc.point_weight[1] = 1.0;

        JointAngleControlInfo ctrl;
        ctrl.constraint_index = static_cast<int>(tetMesh.joint_constraints.size());
        ctrl.axis_dir         = axis_dir;
        ctrl.n_dir            = n_dir;
        ctrl.target_angle     = 0.0;
        ctrl.lower_limit      = -JointAngleControlInfo::kSafeAngleLimit;
        ctrl.upper_limit      =  JointAngleControlInfo::kSafeAngleLimit;
        ctrl.strength_ratio   = 1.0;
        ctrl.joint_name       = "joint_" + std::to_string(i) + "_"
                                + ((i % 2 == 0) ? "Z" : "X");

        tetMesh.joint_constraints.push_back(jc);
        tetMesh.joint_angle_controls.push_back(ctrl);

        tetMesh.collision_exclusion_pairs.push_back({body_ids[i], body_ids[i + 1]});
    }

    // Disable collision between ALL link pairs in the chain.
    // Adjacent pairs are already added above; add all non-adjacent pairs here.
    for(int i = 0; i < NUM_CUBES; i++)
        for(int j = i + 2; j < NUM_CUBES; j++)
            tetMesh.collision_exclusion_pairs.push_back({body_ids[i], body_ids[j]});

    ipc.m_abd_system->parms.joint_strength_ratio            = 100.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 100.0;

    g_joint_control_enabled = true;

    std::cout << "[set_case12] Multi-cube revolute chain loaded (" << NUM_CUBES << " cubes)." << std::endl;
    for(int i = 0; i < NUM_CUBES; i++)
        std::cout << "  cube " << i << ": body_id=" << body_ids[i]
                  << (i == 0 ? " (fixed)" : " (free)") << std::endl;
    std::cout << "[set_case12] Joint constraints: "
              << tetMesh.joint_constraints.size() << std::endl;
    std::cout << "[set_case12] Joint angle controls: "
              << tetMesh.joint_angle_controls.size() << std::endl;
    xRot = 15.0f; yRot = -30.f; yTrans = 0.0f; zTrans = 1.5f;
}

// ==========================================================================
// Case 13: XArm7 with Gripper - Interactive Joint Control via ImGui sliders.
// Same model as case 9, but with UI for controlling each revolute joint angle.
// ==========================================================================
void set_case13_xarm7_interactive()
{
    gipc::UrdfSceneImporter urdf_importer;

    //std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm6_robot_white.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.3;
    global_transform(1, 3) = 0.0;  // place near origin so default camera can see it
    urdf_importer.set_global_transform(global_transform);

    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case13] XArm7 Interactive import failed!" << std::endl;
        std::abort();
    }

    // Keep gravity enabled (rbs-uipc style default).
    // Do NOT override per-vertex apply_gravity or ABD parms.gravity here.

    // Mass-based joint stiffness matching rbs-uipc formulation:
    // kappa = strength_ratio * (m_parent + m_child), NO dt² factor.
    ipc.m_abd_system->parms.joint_strength_ratio            = 100.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 100.0;

    // Enable joint control UI
    g_joint_control_enabled = true;

    std::cout << "[set_case13] XArm7 Interactive loaded." << std::endl;
    std::cout << "[set_case13] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case13] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
    for(auto& ctrl : tetMesh.joint_angle_controls)
    {
        std::cout << "  - " << ctrl.joint_name
                  << " [" << (ctrl.lower_limit * 180.0 / 3.14159265) << ", "
                  << (ctrl.upper_limit * 180.0 / 3.14159265) << "] deg" << std::endl;
    }
    xRot = 15.0f; yRot = -30.f; yTrans = 0.0f; zTrans = 1.5f;
}

// ==========================================================================
// Case 14: XArm7 + Gripper - Interactive Joint Control via ImGui sliders.
// Uses xarm7_with_gripper.urdf (7-DOF arm + gripper fingers).
// ==========================================================================
void set_case14_xarm7_gripper_interactive()
{
    gipc::UrdfSceneImporter urdf_importer;

    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.3;
    global_transform(1, 3) = 0.0;
    urdf_importer.set_global_transform(global_transform);

    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case14] XArm7+Gripper Interactive import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 100.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 100.0;

    ipc.m_skip_all_collision = true;
    g_skip_rendering = false;

    g_joint_control_enabled = true;

    std::cout << "[set_case14] XArm7+Gripper Interactive loaded." << std::endl;
    std::cout << "[set_case14] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case14] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
    for(auto& ctrl : tetMesh.joint_angle_controls)
    {
        std::cout << "  - " << ctrl.joint_name
                  << " [" << (ctrl.lower_limit * 180.0 / 3.14159265) << ", "
                  << (ctrl.upper_limit * 180.0 / 3.14159265) << "] deg" << std::endl;
    }
    xRot = 15.0f; yRot = -30.f; yTrans = 0.0f; zTrans = 1.5f;
}

// ==========================================================================
// Case 15: Ridgeback Dual Panda (nomobile) - Interactive Joint Control.
// Uses ridgeback_dual_panda2_nomobile.urdf (dual 7-DOF arms + grippers).
// ==========================================================================
void set_case15_ridgeback_dual_panda()
{
    gipc::UrdfSceneImporter urdf_importer;

    std::string urdf_path =
        assets_dir + "sim_data/urdf/ridgeback_dual_panda_soft/"
        "franka/ridgeback_dual_panda2_nomobile.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();
    global_transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.3;
    global_transform(1, 3) = 0.0;
    urdf_importer.set_global_transform(global_transform);

    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case15] Ridgeback Dual Panda import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio              = 100.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio   = 100.0;
    ipc.m_abd_system->parms.prismatic_strength_ratio          = 100.0;
    ipc.m_abd_system->parms.prismatic_driving_strength_ratio  = 100.0;

    g_joint_control_enabled = true;

    std::cout << "[set_case15] Ridgeback Dual Panda (nomobile) loaded." << std::endl;
    std::cout << "[set_case15] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case15] Revolute angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
    for(auto& ctrl : tetMesh.joint_angle_controls)
    {
        std::cout << "  - " << ctrl.joint_name
                  << " [" << (ctrl.lower_limit * 180.0 / 3.14159265) << ", "
                  << (ctrl.upper_limit * 180.0 / 3.14159265) << "] deg" << std::endl;
    }
    std::cout << "[set_case15] Prismatic drive controls: " << tetMesh.prismatic_drive_controls.size() << std::endl;
    for(auto& ctrl : tetMesh.prismatic_drive_controls)
    {
        std::cout << "  - " << ctrl.joint_name
                  << " [" << ctrl.lower_limit << ", " << ctrl.upper_limit << "] m" << std::endl;
    }
    xRot = 15.0f; yRot = -30.f; yTrans = 0.0f; zTrans = 1.5f;
}

void set_case16_xarm7_gripper_soft_cube()
{
    // --- ABD first: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;

    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;

    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    Eigen::Matrix4d global_transform = arm_t.matrix();
    urdf_importer.set_global_transform(global_transform);

    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case16] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 1000.0;

    // Mark all robot arm links to skip ground collision
    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }
    std::cout << "[set_case16] " << tetMesh.ground_collision_skip_body_ids.size()
              << " robot bodies will skip ground collision" << std::endl;

    // --- ABD table: fixed rigid body, won't move ---
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;

    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 1;      // X width
        table_tf(1, 1) = 0.02;   // Y thickness
        table_tf(2, 2) = 1;      // Z depth
        table_tf(0, 3) = 0.15;   // X position
        table_tf(1, 3) = -0.79;  // Y position
        table_tf(2, 3) = 0.0;

        table_imp.load_geometry(tetMesh,
                                3,
                                gipc::BodyType::ABD,
                                table_tf,
                                1e9,
                                assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type,
                                BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;  // ABD body right after arm

    // Exclude collision between table and all arm links
    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);

    // Table also skips ground collision
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    std::cout << "[set_case16] Table body_id=" << table_body_id
              << " (Fixed, no arm collision)" << std::endl;

    // --- Cubes on the table (grid layout like set_case6) ---
    bool   cube_use_abd = true;   // true=ABD (rigid), false=FEM (soft)
    double cube_scale   = 0.1;
    double cube_dist    = cube_scale;   // spacing = cube size (touching)
    int    cube_count_x = 1;            // <-- grid columns
    int    cube_count_z = 1;            // <-- grid rows
    double table_cx     = 0.15;         // table center X (same as table_tf)
    double table_cz     = 0.0;          // table center Z
    double table_top_y  = -0.78;        // table top surface Y
    {
        for(int i = 0; i < cube_count_x; i++)
        {
            for(int j = 0; j < cube_count_z; j++)
            {
                double px = table_cx + (i - (cube_count_x - 1) * 0.5) * cube_dist +0.01;
                double pz = table_cz + (j - (cube_count_z - 1) * 0.5) * cube_dist;

                double3 offset = {-px, -(table_top_y + cube_scale * 0.5 - 0.05), -pz};
                Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
                tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cube_scale;
                tf.block<3, 1>(0, 3) = -Eigen::Vector3d(offset.x, offset.y, offset.z);

                gipc::SimpleSceneImporter soft_imp;
                soft_imp.load_geometry(tetMesh,
                                       3,
                                       cube_use_abd ? gipc::BodyType::ABD : gipc::BodyType::FEM,
                                       tf,
                                       cube_use_abd ? 1e8 : 1e3,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type,
                                       BodyBoundaryType::Free);
                std::cout << "[set_case16] Soft cube at (" << px << ", " << pz << ")" << std::endl;
            }
        }
    }
    int num_soft_cubes = cube_count_x * cube_count_z;

    // --- Cloth above table center (configurable resolution) ---
    int    cloth_res   = 15;     // <-- grid resolution: 4~40, gives (res+1)^2 verts
    double cloth_scale = 0.15;  // <-- physical size of the cloth
    double cloth_E     = 1e4;   // <-- Young's modulus (same as case 3)
    {
        std::string cloth_path = generate_cloth_obj(cloth_res);
        gipc::SimpleSceneImporter cloth_imp;
        double3 cloth_offset = {-table_cx, -(table_top_y + 0.1), -table_cz};
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cloth_scale;
        cloth_tf.block<3, 1>(0, 3) = -Eigen::Vector3d(cloth_offset.x, cloth_offset.y, cloth_offset.z);

        cloth_imp.load_geometry(tetMesh,
                                2,
                                gipc::BodyType::FEM,
                                cloth_tf,
                                cloth_E,
                                cloth_path,
                                ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }
    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case16] XArm7+Gripper + Table + " << num_soft_cubes
              << " Cubes + Cloth loaded." << std::endl;
    std::cout << "[set_case16] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case16] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case16] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case16] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
}


void set_case17_xarm7_gripper_cup()
{
    // --- ABD first: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;

    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;

    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case17] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- ABD table: fixed ---
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;
    double table_cx = 0.15, table_cz = 0.0, table_top_y = -0.78;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 1;
        table_tf(1, 1) = 0.02;
        table_tf(2, 2) = 1;
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.79;
        table_tf(2, 3) = table_cz;

        table_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, table_tf,
                                1e9, assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;

    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // Cup mesh bbox: (0.06, 0.01, -0.04) to (0.14, 0.16, 0.04)
    // center_x=0.1, center_z=0, y_min=0.01
    double cup_scale = 0.5;
    std::string cup_msh = assets_dir + "sim_data/tetmesh/softgriper_cup.msh";

    auto make_cup_tf = [&](double cx, double cz) {
        Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
        tf(0, 0) = cup_scale;
        tf(1, 1) = cup_scale;
        tf(2, 2) = cup_scale;
        tf(0, 3) = cx - cup_scale * 0.1;
        tf(1, 3) = table_top_y - cup_scale * 0.01 + 0.02;
        tf(2, 3) = cz;
        return tf;
    };

    bool use_soft_cup = true;  // true=FEM (soft), false=ABD (rigid)
    {
        gipc::SimpleSceneImporter cup_imp;
        Eigen::Matrix4d cup_tf = make_cup_tf(table_cx, table_cz);
        if(use_soft_cup)
        {
            cup_imp.load_geometry(tetMesh, 3, gipc::BodyType::FEM, cup_tf,
                                  1e4, cup_msh, ipc.pcg_data.P_type, BodyBoundaryType::Free);
            std::cout << "[set_case17] FEM cup (soft, E=1e4) loaded" << std::endl;
        }
        else
        {
            cup_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, cup_tf,
                                  1e8, cup_msh, ipc.pcg_data.P_type, BodyBoundaryType::Free);
            std::cout << "[set_case17] ABD cup (rigid) loaded" << std::endl;
        }
    }

    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case17] XArm7+Gripper + Table + 2 Cups loaded." << std::endl;
    std::cout << "[set_case17] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case17] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case17] Total vertices: " << tetMesh.vertexNum << std::endl;
}

void set_case18_table_cloth_no_arm()
{
    // --- ABD table: fixed rigid body ---
    double table_cx = 0.15, table_cz = 0.0, table_top_y = -0.78;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 1;
        table_tf(1, 1) = 0.02;
        table_tf(2, 2) = 1;
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.79;
        table_tf(2, 3) = 0.0;

        table_imp.load_geometry(tetMesh,
                                3,
                                gipc::BodyType::ABD,
                                table_tf,
                                1e9,
                                assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type,
                                BodyBoundaryType::Fixed);
    }
    int table_body_id = tetMesh.abd_fem_count_info.abd_body_num - 1;
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // --- Cube on the table (same as case 16) ---
    bool   cube_use_abd = true;
    double cube_scale   = 0.1;
    double cube_dist    = cube_scale;
    int    cube_count_x = 1;
    int    cube_count_z = 1;
    {
        for(int i = 0; i < cube_count_x; i++)
        {
            for(int j = 0; j < cube_count_z; j++)
            {
                double px = table_cx + (i - (cube_count_x - 1) * 0.5) * cube_dist + 0.01;
                double pz = table_cz + (j - (cube_count_z - 1) * 0.5) * cube_dist;

                double3 offset = {-px, -(table_top_y + cube_scale * 0.5 - 0.05), -pz};
                Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
                tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cube_scale;
                tf.block<3, 1>(0, 3) = -Eigen::Vector3d(offset.x, offset.y, offset.z);

                gipc::SimpleSceneImporter soft_imp;
                soft_imp.load_geometry(tetMesh,
                                       3,
                                       cube_use_abd ? gipc::BodyType::ABD : gipc::BodyType::FEM,
                                       tf,
                                       cube_use_abd ? 1e8 : 1e3,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type,
                                       BodyBoundaryType::Free);
            }
        }
    }
    int num_soft_cubes = cube_count_x * cube_count_z;

    // --- Cloth above table (same as case 16) ---
    int    cloth_res   = 100;
    double cloth_scale = 0.15;
    double cloth_E     = 1e4;
    {
        std::string cloth_path = generate_cloth_obj(cloth_res);
        gipc::SimpleSceneImporter cloth_imp;
        double3 cloth_offset = {-table_cx, -(table_top_y + 0.1), -table_cz};
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cloth_scale;
        cloth_tf.block<3, 1>(0, 3) = -Eigen::Vector3d(cloth_offset.x, cloth_offset.y, cloth_offset.z);

        cloth_imp.load_geometry(tetMesh,
                                2,
                                gipc::BodyType::FEM,
                                cloth_tf,
                                cloth_E,
                                cloth_path,
                                ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    g_joint_control_enabled = false;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case18] Table + " << num_soft_cubes
              << " Cubes + Cloth (NO robot arm) loaded." << std::endl;
    std::cout << "[set_case18] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case18] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case18] Total vertices: " << tetMesh.vertexNum << std::endl;
}

void set_case19_arm_table_cloth()
{
    // --- ABD first: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;
    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case19] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- ABD table: fixed rigid body ---
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;
    double table_cx = 0.15, table_cz = 0.0, table_top_y = -0.78;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 1;
        table_tf(1, 1) = 0.02;
        table_tf(2, 2) = 1;
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.79;
        table_tf(2, 3) = 0.0;

        table_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, table_tf,
                                1e9, assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;
    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // --- Cube on the table ---
    bool   cube_use_abd = true;
    double cube_scale   = 0.1;
    double cube_dist    = cube_scale;
    int    cube_count_x = 1;
    int    cube_count_z = 1;
    {
        for(int i = 0; i < cube_count_x; i++)
            for(int j = 0; j < cube_count_z; j++)
            {
                double px = table_cx + (i - (cube_count_x - 1) * 0.5) * cube_dist + 0.01;
                double pz = table_cz + (j - (cube_count_z - 1) * 0.5) * cube_dist;
                double3 offset = {-px, -(table_top_y + cube_scale * 0.5 - 0.05), -pz};
                Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
                tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cube_scale;
                tf.block<3, 1>(0, 3) = -Eigen::Vector3d(offset.x, offset.y, offset.z);

                gipc::SimpleSceneImporter soft_imp;
                soft_imp.load_geometry(tetMesh, 3,
                                       cube_use_abd ? gipc::BodyType::ABD : gipc::BodyType::FEM,
                                       tf, cube_use_abd ? 1e8 : 1e3,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type, BodyBoundaryType::Free);
            }
    }
    int num_soft_cubes = cube_count_x * cube_count_z;

    // --- Cloth above table (same params as case 3) ---
    int    cloth_res   = 15;
    double cloth_scale = 0.15;
    double cloth_E     = 1e4;
    {
        std::string cloth_path = generate_cloth_obj(cloth_res);
        gipc::SimpleSceneImporter cloth_imp;
        double3 cloth_offset = {-table_cx, -(table_top_y + 0.1), -table_cz};
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cloth_scale;
        cloth_tf.block<3, 1>(0, 3) = -Eigen::Vector3d(cloth_offset.x, cloth_offset.y, cloth_offset.z);

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, cloth_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case19] Arm + Table + " << num_soft_cubes
              << " Cubes + Cloth loaded." << std::endl;
    std::cout << "[set_case19] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case19] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case19] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case19] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
}

void set_case20_arm_hanging_cloth()
{
    // --- ABD first: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;
    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case20] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- FEM cloth: case 3 style (cloth_high.obj, fixed at top corners) ---
    // Positioned near the arm with X offset so the arm can reach it
    ipc.clothThickness = 1e-3;
    int fem_vert_start = tetMesh.vertexNum;
    {
        gipc::SimpleSceneImporter importer;
        double scale = 0.4;
        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0.5, -0.3, 0});
        t.scale(scale);
        t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));

        std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
        importer.load_geometry(tetMesh, 2, gipc::BodyType::FEM, t.matrix(),
                               1e4, mesh_path, ipc.pcg_data.P_type);
    }

    // Fix top-corner vertices (same logic as case 3, but only over FEM verts)
    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = fem_vert_start; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "[set_case20] fixed vertex num: " << fixed_vertex_num << std::endl;

    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case20] Arm + Hanging Cloth loaded." << std::endl;
    std::cout << "[set_case20] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case20] FEM cloth verts: " << (tetMesh.vertexNum - fem_vert_start) << std::endl;
    std::cout << "[set_case20] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case20] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
}

void set_case21_arm_hanging_cloth_configurable()
{
    // --- ABD first: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;
    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case21] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- FEM cloth: configurable resolution, fixed at top corners ---
    int    cloth_res   = 100;    // <-- resolution: (res+1)^2 verts, e.g. 60 -> 3721 verts
    double cloth_scale = 0.8;   // <-- physical size (generate_cloth_obj spans [-0.5,0.5], cloth_high spans [-1,1], so 2x scale)
    double cloth_E     = 1e4;   // <-- Young's modulus (same as case 3)
    ipc.clothThickness = 1e-3;

    int fem_vert_start = tetMesh.vertexNum;
    {
        std::string cloth_path = generate_cloth_obj(cloth_res);
        gipc::SimpleSceneImporter cloth_imp;

        Transform t = Transform::Identity();
        t.translate(Eigen::Vector3d{0.5, -0.3, 0});
        t.scale(cloth_scale);
        t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, t.matrix(),
                                cloth_E, cloth_path, ipc.pcg_data.P_type);
    }

    // Fix top-corner vertices (same logic as case 3/20, only over FEM verts)
    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = fem_vert_start; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "[set_case21] fixed vertex num: " << fixed_vertex_num << std::endl;

    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "[set_case21] Arm + Hanging Cloth (res=" << cloth_res
              << ", " << ((cloth_res+1)*(cloth_res+1)) << " verts) loaded." << std::endl;
    std::cout << "[set_case21] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case21] FEM cloth verts: " << (tetMesh.vertexNum - fem_vert_start) << std::endl;
    std::cout << "[set_case21] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case21] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
}

void set_case22_franka_table_cloth()
{
    // --- ABD: load Franka Panda via URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/franka_panda/panda_arm_hand.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    // Franka URDF is in Z-up metres; rotate -90° about X for Y-up
    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.71, 0));
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case22] Franka Panda import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio  = 1000.0;
    ipc.m_abd_system->parms.prismatic_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- ABD table: 0.8 x 0.2 x 0.8 m (matching Isaac Lab cube 0.2 * scale(4,4,1)) ---
    // cube.msh range is 0.4 per axis, so scale = desired_size / 0.4
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;
    double table_cx = 0.55, table_cz = 0.0;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 2.0;    // 2.0 * 0.4 = 0.8m wide
        table_tf(1, 1) = 0.5;    // 0.5 * 0.4 = 0.2m thick
        table_tf(2, 2) = 2.0;    // 2.0 * 0.4 = 0.8m deep
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.62;  // top = 0.5*0.5 + (-0.62) = table_top_y
        table_tf(2, 3) = table_cz;

        table_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, table_tf,
                                1e9, assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;
    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // --- FEM shirt: T-shirt mesh from Isaac Lab ---
    // shirt_831v.obj Y-bounds: [-0.136, 0.138]. Place so bottom is 0.084m
    // above table top (matching Isaac Lab gap): centre Y = -0.15
    double cloth_E              = 1e4;    // Young's modulus passed to load_geometry
    ipc.clothThickness          = 1e-3;   // shell thickness (m)
    ipc.clothYoungModulus       = 1e4;    // in-plane Young's modulus (stretch)
    // ipc.bendYoungModulus        = 1e5;    // bending Young's modulus
    ipc.clothDensity            = 1000;    // density (kg/m^3), heavier → flattens more
    ipc.strainRate              = 100;      // shear stiffness multiplier
    ipc.bendStiff               = 1e-5;   // bending stiffness
    ipc.softMotionRate          = 1e0;    // softbody motion damping
    ipc.PoissonRate             = 0.49;   // Poisson's ratio
    ipc.gd_frictionRate         = 0.4;    // ground friction
    ipc.frictionRate            = 0.4;    // contact friction
    ipc.relative_dhat           = 1e-3;   // relative collision thickness, smaller → layers closer
    ipc.IPC_dt                  = 1e-2;   // timestep (s)

    {
        std::string shirt_path = assets_dir + "triMesh/shirt_831v.obj";
        gipc::SimpleSceneImporter cloth_imp;
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf(0, 3) = table_cx;
        cloth_tf(1, 3) = -0.15;   // bottom=-0.286, gap to table top(-0.37)=0.084m
        cloth_tf(2, 3) = table_cz;

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, shirt_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    // --- Trajectory playback ---
    g_joint_control_enabled        = true;
    g_trajectory_playback_enabled  = true;
    g_trajectory_sim_time          = 0.0;
    g_skip_rendering               = false;

    std::string traj_path = assets_dir + "trajectories/franka_fold.txt";
    if(!load_trajectory(traj_path))
        std::cerr << "[set_case22] WARNING: no trajectory file at " << traj_path << std::endl;

    g_settling_frames = 10;

    std::cout << "[set_case22] Franka Panda + Table + Shirt loaded." << std::endl;
    std::cout << "[set_case22] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case22] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case22] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case22] Joint controls (revolute): " << tetMesh.joint_angle_controls.size() << std::endl;
    std::cout << "[set_case22] Joint controls (prismatic): " << tetMesh.prismatic_drive_controls.size() << std::endl;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;
}

void set_case23_xarm_table_cloth()
{
    // --- ABD: load XArm7+Gripper via URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/xarm/xarm7_with_gripper.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    double arm_scale = 0.3;

    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.75, 0));
    arm_t.scale(arm_scale);
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case23] XArm7+Gripper import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio  = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- ABD table: fixed rigid body ---
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;
    double table_cx = 0.15, table_cz = 0.0, table_top_y = -0.78;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 1;
        table_tf(1, 1) = 0.02;
        table_tf(2, 2) = 1;
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.79;
        table_tf(2, 3) = table_cz;

        table_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, table_tf,
                                1e9, assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;
    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // --- FEM cloth: flat on table ---
    int    cloth_res   = 30;
    double cloth_scale = 0.15;
    double cloth_E     = 1e4;
    ipc.clothThickness = 1e-3;

    {
        std::string cloth_path = generate_cloth_obj(cloth_res);
        gipc::SimpleSceneImporter cloth_imp;
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * cloth_scale;
        cloth_tf(0, 3) = table_cx;
        cloth_tf(1, 3) = table_top_y + 0.02;
        cloth_tf(2, 3) = table_cz;

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, cloth_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    // --- Trajectory playback ---
    g_joint_control_enabled        = true;
    g_trajectory_playback_enabled  = true;
    g_trajectory_sim_time          = 0.0;
    g_skip_rendering               = false;

    std::string traj_path = assets_dir + "trajectories/xarm7_fold.txt";
    if(!load_trajectory(traj_path))
        std::cerr << "[set_case23] WARNING: no trajectory file at " << traj_path << std::endl;

    g_settling_frames = 10;

    std::cout << "[set_case23] XArm7 + Table + Cloth loaded." << std::endl;
    std::cout << "[set_case23] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case23] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case23] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case23] Joint controls (revolute): " << tetMesh.joint_angle_controls.size() << std::endl;
    std::cout << "[set_case23] Joint controls (prismatic): " << tetMesh.prismatic_drive_controls.size() << std::endl;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;
}

void set_case24_franka_coarse()
{
    // --- ABD: load Franka Panda via coarse URDF ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path = assets_dir + "sim_data/urdf/franka_panda/panda_arm_hand_coarse.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(0, -0.71, 0));
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case24] Franka Panda (coarse) import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio  = 1000.0;
    ipc.m_abd_system->parms.prismatic_driving_strength_ratio = 1000.0;

    for(auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(link_info.body_id);
    }

    // --- ABD table ---
    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;
    double table_cx = 0.55, table_cz = 0.0;
    {
        gipc::SimpleSceneImporter table_imp;
        Eigen::Matrix4d table_tf = Eigen::Matrix4d::Identity();
        table_tf(0, 0) = 2.0;
        table_tf(1, 1) = 0.5;
        table_tf(2, 2) = 2.0;
        table_tf(0, 3) = table_cx;
        table_tf(1, 3) = -0.62;
        table_tf(2, 3) = table_cz;

        table_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, table_tf,
                                1e9, assets_dir + "tetMesh/cube.msh",
                                ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int table_body_id = arm_body_count;
    for(int arm_id = 0; arm_id < arm_body_count; arm_id++)
        tetMesh.collision_exclusion_pairs.emplace_back(table_body_id, arm_id);
    tetMesh.ground_collision_skip_body_ids.push_back(table_body_id);

    // --- FEM shirt ---
    double cloth_E              = 1e4;
    ipc.clothThickness          = 1e-3;
    ipc.clothYoungModulus       = 1e4;
    ipc.clothDensity            = 1000;
    ipc.strainRate              = 100;
    ipc.bendStiff               = 1e-5;
    ipc.softMotionRate          = 1e0;
    ipc.PoissonRate             = 0.49;
    ipc.gd_frictionRate         = 0.4;
    ipc.frictionRate            = 0.4;
    ipc.relative_dhat           = 1e-3;
    ipc.IPC_dt                  = 1e-2;

    {
        std::string shirt_path = assets_dir + "triMesh/shirt_831v.obj";
        gipc::SimpleSceneImporter cloth_imp;
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();
        cloth_tf(0, 3) = table_cx;
        cloth_tf(1, 3) = -0.15;
        cloth_tf(2, 3) = table_cz;

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, shirt_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    // --- Trajectory playback ---
    g_joint_control_enabled        = true;
    g_trajectory_playback_enabled  = true;
    g_trajectory_sim_time          = 0.0;
    g_skip_rendering               = false;

    std::string traj_path = assets_dir + "trajectories/franka_fold.txt";
    if(!load_trajectory(traj_path))
        std::cerr << "[set_case24] WARNING: no trajectory file at " << traj_path << std::endl;

    g_settling_frames = 10;

    std::cout << "[set_case24] Franka Panda (coarse) + Table + Shirt loaded." << std::endl;
    std::cout << "[set_case24] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case24] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case24] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case24] Joint controls (revolute): " << tetMesh.joint_angle_controls.size() << std::endl;
    std::cout << "[set_case24] Joint controls (prismatic): " << tetMesh.prismatic_drive_controls.size() << std::endl;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;
}

void set_case25_shirt_freefall()
{
    // true  = Isaac Lab cloth parameters (from rbs_franka_cloth_grasp-onlyCloth_params.txt)
    // false = Stiff-GIPC default cloth parameters (same as case 4 curtain)
    constexpr bool use_isaaclab_params = false;

    double cloth_E;
    if (use_isaaclab_params)
    {
        cloth_E                     = 1e4;
        ipc.clothThickness          = 5e-4;    // 0.5 mm
        ipc.clothYoungModulus       = 1e4;     // 10 kPa
        ipc.bendYoungModulus        = 1e4;
        ipc.clothDensity            = 20.0;    // 20 kg/m^3
        ipc.strainRate              = 1;
        ipc.bendStiff               = 10.0;    // blendingStiffness = 10
        ipc.softMotionRate          = 1e0;
        ipc.PoissonRate             = 0.499;
        ipc.gd_frictionRate         = 0.5;
        ipc.frictionRate            = 0.5;
        ipc.relative_dhat           = 1e-3;    // d_hat = 1 mm
        ipc.IPC_dt                  = 1e-2;    // 100 Hz
    }
    else
    {
        cloth_E                     = 1e4;
        ipc.clothThickness          = 1e-3;    // 1 mm
        ipc.clothYoungModulus       = 1e6;
        ipc.bendYoungModulus        = 1e5;
        ipc.clothDensity            = 2e2;     // 200 kg/m^3
        ipc.strainRate              = 100;
        ipc.bendStiff               = 3e-4;
        ipc.softMotionRate          = 1e0;
        ipc.PoissonRate             = 0.49;
        ipc.gd_frictionRate         = 0.4;
        ipc.frictionRate            = 0.4;
        ipc.relative_dhat           = 1e-3;
        ipc.IPC_dt                  = 1e-2;
    }

    // Full Newton convergence (no semi-implicit exit)
    ipc.semi_implicit_enabled = false;

    // No table — shirt falls directly onto the built-in ground plane (Y = -1).
    {
        std::string shirt_path = assets_dir + "triMesh/shirt_6436v.obj";
        gipc::SimpleSceneImporter cloth_imp;
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, shirt_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    g_skip_rendering       = false;
    g_headless_benchmark   = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "=== Case25: SHIRT FREE-FALL (full Newton, params="
              << (use_isaaclab_params ? "IsaacLab" : "StiffGIPC-default") << ") ===" << std::endl;
    std::cout << "FEM bodies: "  << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "Total verts: " << tetMesh.vertexNum << std::endl;
    std::cout << "clothYoungModulus: " << ipc.clothYoungModulus << std::endl;
    std::cout << "clothDensity:     " << ipc.clothDensity << std::endl;
    std::cout << "clothThickness:   " << ipc.clothThickness << std::endl;
    std::cout << "bendStiff:        " << ipc.bendStiff << std::endl;
    std::cout << "PoissonRate:      " << ipc.PoissonRate << std::endl;
    std::cout << "relative_dhat:    " << ipc.relative_dhat << std::endl;
    std::cout << "IPC_dt:           " << ipc.IPC_dt << std::endl;
    std::cout << "semi_implicit:    OFF (full Newton)" << std::endl;
}

void set_case26_shirt_freefall_semi_implicit()
{
    double cloth_E                  = 1e4;
    ipc.clothThickness              = 1e-3;    // 1 mm
    ipc.clothYoungModulus           = 1e6;
    ipc.bendYoungModulus            = 1e5;
    ipc.clothDensity                = 2e2;     // 200 kg/m^3
    ipc.strainRate                  = 100;
    ipc.bendStiff                   = 3e-4;
    ipc.softMotionRate              = 1e0;
    ipc.PoissonRate                 = 0.49;
    ipc.gd_frictionRate             = 0.4;
    ipc.frictionRate                = 0.4;
    ipc.relative_dhat               = 1e-3;
    ipc.IPC_dt                      = 1e-2;

    // Enable semi-implicit beta-decay early exit
    ipc.semi_implicit_enabled  = true;
    ipc.semi_implicit_beta_tol = 1e-3;
    ipc.semi_implicit_min_iter = 1;

    // No table — falls to built-in ground plane (Y = -1)
    {
        std::string shirt_path = assets_dir + "triMesh/shirt_6436v.obj";
        gipc::SimpleSceneImporter cloth_imp;
        Eigen::Matrix4d cloth_tf = Eigen::Matrix4d::Identity();

        cloth_imp.load_geometry(tetMesh, 2, gipc::BodyType::FEM, cloth_tf,
                                cloth_E, shirt_path, ipc.pcg_data.P_type,
                                BodyBoundaryType::Free);
    }

    g_skip_rendering       = false;
    g_headless_benchmark   = false;
    xRot = 5.0f; yRot = -45.f; yTrans = 0.5f; zTrans = 2.0f;

    std::cout << "=== Case26: SHIRT FREE-FALL (semi-implicit, StiffGIPC-default) ===" << std::endl;
    std::cout << "FEM bodies: "  << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "Total verts: " << tetMesh.vertexNum << std::endl;
    std::cout << "clothYoungModulus: " << ipc.clothYoungModulus << std::endl;
    std::cout << "clothDensity:     " << ipc.clothDensity << std::endl;
    std::cout << "clothThickness:   " << ipc.clothThickness << std::endl;
    std::cout << "bendStiff:        " << ipc.bendStiff << std::endl;
    std::cout << "PoissonRate:      " << ipc.PoissonRate << std::endl;
    std::cout << "relative_dhat:    " << ipc.relative_dhat << std::endl;
    std::cout << "IPC_dt:           " << ipc.IPC_dt << std::endl;
    std::cout << "semi_implicit:    ON (beta_tol=" << ipc.semi_implicit_beta_tol
              << ", min_iter=" << ipc.semi_implicit_min_iter << ")" << std::endl;
}

// ==========================================================================
// Case 27: RealMan arm + box + roller + FEM soft cube
// Reproduces realman_soft_cube_usd_test.py for C++ debugging.
// Semi-implicit OFF by default to investigate Newton convergence issues.
// ==========================================================================
void set_case27_realman_soft_cube()
{
    // --- 1. RealMan arm via URDF (ABD) ---
    gipc::UrdfSceneImporter urdf_importer;
    std::string urdf_path =
        "/home/ps/Downloads/robot_models/realman_03_description/"
        "xhand_with_realman_arm_only_obb.urdf";
    urdf_importer.set_urdf_path(urdf_path);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;

    // Y-up world: Rx(-90) converts URDF Z-up to Y-up,
    // then Rz(-90) matches the Python scene's Quat(0.707, 0, 0, -0.707).
    Transform arm_t = Transform::Identity();
    arm_t.translate(Eigen::Vector3d(-0.3, 0.05, 0.0));
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));
    arm_t.rotate(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ()));

    urdf_importer.set_global_transform(arm_t.matrix());
    urdf_importer.set_root_fixed(true);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(false);
    urdf_importer.set_default_young_modulus(1e7);

    // Isaac Lab initial joint angles (radians) — applied during FK so arm
    // is loaded already at the target pose, avoiding pass-through collisions.
    std::map<std::string, double> init_angles = {
        {"RH_joint1",  1.2},
        {"RH_joint2", -0.5411},
        {"RH_joint3",  0.3002},
        {"RH_joint4",  1.3},
        {"RH_joint5", -2.2951},
        {"RH_joint6",  0.0},
    };
    urdf_importer.set_initial_joint_angles(init_angles);

    bool success = urdf_importer.import_scene(tetMesh, ipc.pcg_data.P_type);
    if(!success)
    {
        std::cerr << "[set_case27] RealMan arm import failed!" << std::endl;
        std::abort();
    }

    ipc.m_abd_system->parms.joint_strength_ratio            = 1000.0;
    ipc.m_abd_system->parms.revolute_driving_strength_ratio  = 1000.0;

    // Skip ground collision for all robot links
    for(auto& [name, info] : urdf_importer.link_infos())
        if(info.body_id >= 0)
            tetMesh.ground_collision_skip_body_ids.push_back(info.body_id);

    int arm_body_count = tetMesh.abd_fem_count_info.abd_body_num;

    // The importer already sets target_angle = initial_angle and
    // initial_angle_offset = initial_angle for joints in init_angles,
    // so the driving energy correctly holds the FK pose (effective target = 0).

    // --- 2. Box (ABD, Fixed/Kinematic) ---
    // cube.msh: each axis spans 0.4 units. Target box: ~0.43 x 0.21 x 0.65 m
    // In Y-up: box center at Y=1.0 (was Z=1.0 in Python Z-up)
    {
        gipc::SimpleSceneImporter box_imp;
        Eigen::Matrix4d box_tf = Eigen::Matrix4d::Identity();
        box_tf(0, 0) = 0.43 / 0.4;   // X scale
        box_tf(1, 1) = 0.21 / 0.4;   // Y scale (short axis = opening depth)
        box_tf(2, 2) = 0.65 / 0.4;   // Z scale (tall axis)
        box_tf(0, 3) = 0.0;
        box_tf(1, 3) = 1.0;           // Y position
        box_tf(2, 3) = 0.0;

        box_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, box_tf,
                              1e9, assets_dir + "tetMesh/cube.msh",
                              ipc.pcg_data.P_type, BodyBoundaryType::Fixed);
    }
    int box_body_id = arm_body_count;

    // Exclude collision between box and all arm links
    for(int i = 0; i < arm_body_count; i++)
        tetMesh.collision_exclusion_pairs.emplace_back(box_body_id, i);
    tetMesh.ground_collision_skip_body_ids.push_back(box_body_id);

    // --- 3. MINI_ROLLER stand-in (ABD, Free) ---
    // Python: MINI_ROLLER OBB * scale 0.75 = (0.069, 0.131, 0.125) m
    // cube.msh spans 0.4 per axis, so scale = extents / 0.4
    // Z-up -> Y-up: Z_zup(0.125) -> Y_yup, Y_zup(0.131) -> Z_yup
    {
        gipc::SimpleSceneImporter roller_imp;
        Eigen::Matrix4d roller_tf = Eigen::Matrix4d::Identity();
        roller_tf(0, 0) = 0.069 / 0.4;   // X: ~0.172
        roller_tf(1, 1) = 0.125 / 0.4;   // Y (up): ~0.312
        roller_tf(2, 2) = 0.131 / 0.4;   // Z: ~0.327
        roller_tf(0, 3) = 0.0;
        roller_tf(1, 3) = 1.5;   // above box
        roller_tf(2, 3) = 0.0;

        roller_imp.load_geometry(tetMesh, 3, gipc::BodyType::ABD, roller_tf,
                                 1e8, assets_dir + "tetMesh/cube.msh",
                                 ipc.pcg_data.P_type, BodyBoundaryType::Free);
    }

    // --- 4. FEM Soft Cube (must be after all ABD) ---
    {
        gipc::SimpleSceneImporter soft_imp;
        double cube_scale = 0.15;   // 0.4 * 0.15 = 0.06 m = 6 cm
        Eigen::Matrix4d soft_tf = Eigen::Matrix4d::Identity();
        soft_tf(0, 0) = cube_scale;
        soft_tf(1, 1) = cube_scale;
        soft_tf(2, 2) = cube_scale;
        soft_tf(0, 3) = 0.0;
        soft_tf(1, 3) = 2.5;     // well above box
        soft_tf(2, 3) = 0.0;

        soft_imp.load_geometry(tetMesh, 3, gipc::BodyType::FEM, soft_tf,
                               1e5, assets_dir + "tetMesh/cube.msh",
                               ipc.pcg_data.P_type, BodyBoundaryType::Free);
    }

    // Physics parameters (matching Python Config)
    ipc.IPC_dt        = 0.01;
    ipc.YoungModulus   = 1e7;
    ipc.frictionRate   = 0.4;
    ipc.gd_frictionRate = 0.4;
    ipc.relative_dhat  = 1e-3;

    // Semi-implicit ON to match Python scene (needed to reproduce jitter bug)
    ipc.semi_implicit_enabled = true;

    g_joint_control_enabled = true;
    g_skip_rendering = false;
    xRot = 25.0f; yRot = -45.f; yTrans = -0.8f; zTrans = 2.0f;

    std::cout << "[set_case27] RealMan + Box + Roller + Soft Cube loaded." << std::endl;
    std::cout << "[set_case27] ABD bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "[set_case27] FEM bodies: " << tetMesh.abd_fem_count_info.fem_body_num << std::endl;
    std::cout << "[set_case27] Total vertices: " << tetMesh.vertexNum << std::endl;
    std::cout << "[set_case27] Joint angle controls: " << tetMesh.joint_angle_controls.size() << std::endl;
    std::cout << "[set_case27] semi_implicit: " << (ipc.semi_implicit_enabled ? "ON" : "OFF") << std::endl;
}

// ==========================================================================
// ABD Freefall Benchmark: load xarm6 STL meshes as independent ABD bodies
// (no joints, no collision, no rendering, headless)
// ==========================================================================
void set_case_abd_freefall_benchmark()
{
    std::string stl_dir = assets_dir + "sim_data/urdf/xarm/xarm_description/meshes/xarm6/visual/";
    std::vector<std::string> stl_files = {
        stl_dir + "base.stl",
        stl_dir + "link1.stl",
        stl_dir + "link2.stl",
        stl_dir + "link3.stl",
        stl_dir + "link4.stl",
        stl_dir + "link5.stl",
        stl_dir + "link6.stl"
    };

    double scale = 0.3;
    double young_modulus = 1e7;

    for(size_t i = 0; i < stl_files.size(); i++)
    {
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
        transform(1, 3) = 0.5 + i * 0.15;

        bool ok = tetMesh.load_surfaceMesh_ABD(stl_files[i],
                                                transform,
                                                young_modulus,
                                                BodyBoundaryType::Free);
        if(!ok)
        {
            std::cerr << "[benchmark] Failed to load: " << stl_files[i] << std::endl;
        }
    }

    // Benchmark mode: prioritize throughput over accuracy
    ipc.Newton_solver_threshold = 5e-2;
    ipc.pcg_threshold           = 1e-3;
    ipc.frictionRate            = 0.0;
    ipc.gd_frictionRate         = 0.0;

    g_skip_rendering = true;
    g_headless_benchmark = true;
    ipc.m_skip_all_collision = true;
    saveSurface = true;

    std::cout << "=== ABD FREEFALL BENCHMARK ===" << std::endl;
    std::cout << "Bodies: " << tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "Total verts: " << tetMesh.vertexNum << std::endl;
    std::cout << "Rendering: DISABLED" << std::endl;
    std::cout << "Collision: DISABLED" << std::endl;
    std::cout << "OpenGL window: HIDDEN" << std::endl;
    std::cout << "OBJ export: ENABLED" << std::endl;
    std::cout << "==============================" << std::endl;
}

void setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(int i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = i;
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
            {
                index = 0;
            }
        }
    }
    index = 0;
    for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
    {

        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = i;
            index++;
        }
    }
}

void initScene()
{
    std::filesystem::exists(metis_dir) || std::filesystem::create_directory(metis_dir);
    ipc.pcg_data.P_type = 1;

    int scene_no            = g_scene_no;
    g_joint_control_enabled = false;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!ABD must be loaded before FEM!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    switch(scene_no)
    {
        case 0:  // box pipe
            set_case1();
            break;
        case 1:  // soft-rigid-cloth coupling
            set_case2();
            break;
        case 2:  //wrecking ball case
            set_case3();
            break;
        case 3:  //fixed cloth
            set_case4();
            break;
        case 4:  //twisting mat
            set_case5();
            break;
        case 5:  //box pipe large scale and cloth
            set_case6();
            break;
        case 6:  //URDF import test
            set_case7_urdf_test(); 
            break;
        case 7:  //XArm6 URDF test
            set_case8_xarm6_test();
            break;
        case 8:  //XArm7 + Gripper URDF test
            set_case9_xarm7_gripper_test();
            break;
        case 9:  //UIPC-style soft gripper demo
            set_case10_uipc_demo();
            break;
        case 10: //Full soft gripper scene
            set_case11_gripper();
            break;
        case 11: //Case12: two ABD cubes revolute test with UI slider
            set_case12_two_cube_revolute_test();
            break;
        case 12: //Case13: XArm6 interactive joint control
            set_case13_xarm7_interactive();
            break;
        case 13: //Case14: XArm7 + Gripper interactive joint control
            set_case14_xarm7_gripper_interactive();
            break;
        case 14: //Case15: Ridgeback Dual Panda (nomobile)
            set_case15_ridgeback_dual_panda();
            break;
        case 15: //Case16: XArm7 + Gripper + soft cube + table + cloth
            set_case16_xarm7_gripper_soft_cube();
            break;
        case 16: //Case17: XArm7 + Gripper + table + cup
            set_case17_xarm7_gripper_cup();
            break;
        case 17: //Case18: Table + cloth (no arm)
            set_case18_table_cloth_no_arm();
            break;
        case 18: //Case19: Arm + table + cube + cloth
            set_case19_arm_table_cloth();
            break;
        case 19: //Case20: Arm + hanging cloth
            set_case20_arm_hanging_cloth();
            break;
        case 20: //Case21: Arm + hanging cloth (configurable)
            set_case21_arm_hanging_cloth_configurable();
            break;
        case 21: //Case22: Franka Panda + table + shirt + trajectory
            set_case22_franka_table_cloth();
            break;
        case 22: //Case23: XArm7 + table + cloth + trajectory
            set_case23_xarm_table_cloth();
            break;
        case 23: //Case24: Franka coarse + table + shirt + trajectory
            set_case24_franka_coarse();
            break;
        case 24: //Case25: Shirt free-fall full Newton
            set_case25_shirt_freefall();
            break;
        case 25: //Case26: Shirt free-fall semi-implicit
            set_case26_shirt_freefall_semi_implicit();
            break;
        case 26: //Case27: RealMan + box + roller + soft cube (Newton debug)
            set_case27_realman_soft_cube();
            break;
        case 99: //ABD Freefall benchmark (no joints, no render, headless)
            set_case_abd_freefall_benchmark();
            break;
    }


    setMAS_partition();


    tetMesh.getSurface();

    initFEM(tetMesh);
    //device_TetraData d_tetMesh;
    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                tetMesh.tri_edges.size(),
                                tetMesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses,
                              tetMesh.masses.data(),
                              tetMesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.apply_gravity,
                              tetMesh.apply_gravity.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.lengthRate,
                              tetMesh.lengthRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volumeRate,
                              tetMesh.volumeRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum,
                              tetMesh.volum.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras,
                              tetMesh.tetrahedras.data(),
                              tetMesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses,
                              tetMesh.DM_inverse.data(),
                              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType,
                              tetMesh.boundaryTypies.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities,
                              tetMesh.velocities.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex,
                              tetMesh.targetIndex.data(),
                              tetMesh.softNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                              tetMesh.targetPos.data(),
                              tetMesh.softNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triDmInverses,
                              tetMesh.tri_DM_inverse.data(),
                              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.area,
                              tetMesh.area.data(),
                              tetMesh.triangleNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triangles,
                              tetMesh.triangles.data(),
                              tetMesh.triangleNum * sizeof(uint3),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edges,
                              tetMesh.tri_edges.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edge_adj_vertex,
                              tetMesh.tri_edges_adj_points.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_id_to_boundary_type,
                              tetMesh.body_id_to_is_fixed.data(),
                              tetMesh.body_id_to_is_fixed.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.point_id_to_body_id,
                              tetMesh.point_id_to_body_id.data(),
                              tetMesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tet_id_to_body_id,
                              tetMesh.tet_id_to_body_id.data(),
                              tetMesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    // Upload per-body motor parameters
    if(!tetMesh.body_motor_infos.empty())
    {
        int bodyNum = static_cast<int>(tetMesh.body_motor_infos.size());
        std::vector<double> motor_params(bodyNum * 5, 0.0);
        for(int i = 0; i < bodyNum; i++)
        {
            auto& mi             = tetMesh.body_motor_infos[i];
            motor_params[i * 5 + 0] = mi.axis_x;
            motor_params[i * 5 + 1] = mi.axis_y;
            motor_params[i * 5 + 2] = mi.axis_z;
            motor_params[i * 5 + 3] = mi.speed;
            motor_params[i * 5 + 4] = mi.strength;
        }
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_motor_params,
                                  motor_params.data(),
                                  bodyNum * 5 * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }

    // Build and upload collision exclusion matrix from host pairs
    if(!tetMesh.collision_exclusion_pairs.empty() && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_matrix(N * N, 0);
        for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
        {
            if(a >= 0 && a < N && b >= 0 && b < N)
            {
                host_matrix[a * N + b] = 1;
                host_matrix[b * N + a] = 1;  // symmetric
            }
        }
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.collision_skip_matrix,
                                  host_matrix.data(),
                                  N * N * sizeof(int),
                                  cudaMemcpyHostToDevice));
        printf("[CollisionExclusion] Uploaded %dx%d exclusion matrix (%d pairs)\n",
               N, N, (int)tetMesh.collision_exclusion_pairs.size());
    }

    // Upload per-body ground collision skip flags
    if(!tetMesh.ground_collision_skip_body_ids.empty() && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_flags(N, 0);
        for(int bid : tetMesh.ground_collision_skip_body_ids)
        {
            if(bid >= 0 && bid < N)
                host_flags[bid] = 1;
        }
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.ground_skip_body,
                                  host_flags.data(),
                                  N * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ipc._ground_skip_body  = d_tetMesh.ground_skip_body;
        ipc._ground_body_count = N;
        printf("[GroundExclusion] %d bodies skip ground collision\n",
               (int)tetMesh.ground_collision_skip_body_ids.size());
    }

    printf("stretchStiff:  %f,  shearStiff:   %f\n", ipc.stretchStiff, ipc.shearStiff);

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = tetMesh.surfVerts.size();
    ipc.surface_Num    = tetMesh.surface.size();
    ipc.edge_Num       = tetMesh.surfEdges.size();
    ipc.tri_edge_num   = tetMesh.tri_edges.size();

    if(ipc.m_skip_all_collision)
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM = 1;
        ipc.MAX_COLLITION_PAIRS_NUM = 1;
    }
    else
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM =
            1 * collision_detection_buff_scale
            * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
               * std::max((ipc.IPC_dt / 0.01), 2.0));
        ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                                      * 3 * collision_detection_buff_scale;
    }

    ipc.triangleNum        = tetMesh.triangleNum;
    ipc.targetVert         = d_tetMesh.targetVert;
    ipc.targetInd          = d_tetMesh.targetIndex;
    ipc.softNum            = tetMesh.softNum;

    // Upload stitch spring data to GPU and set bilateral coupling pointers
    if(tetMesh.softNum > 0 && !d_tetMesh.stitch_paired_vertex.empty())
    {
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.d_stitch_paired_vertex,
                                  d_tetMesh.stitch_paired_vertex.data(),
                                  tetMesh.softNum * sizeof(int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.d_stitch_rest_offset,
                                  d_tetMesh.stitch_rest_offset.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.d_stitch_abd_body_id,
                                  d_tetMesh.stitch_abd_body_id.data(),
                                  tetMesh.softNum * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ipc.m_d_stitch_paired_vertex = d_tetMesh.d_stitch_paired_vertex;
        ipc.m_d_stitch_rest_offset   = d_tetMesh.d_stitch_rest_offset;
        ipc.m_d_stitch_abd_body_id   = d_tetMesh.d_stitch_abd_body_id;
        std::cout << "[stitch] Uploaded " << tetMesh.softNum
                  << " bilateral stitch springs to GPU" << std::endl;
    }

    ipc.abd_fem_count_info = tetMesh.abd_fem_count_info;
    ipc.num_joint_constraints = static_cast<int>(tetMesh.joint_constraints.size());

    std::cout << "ABD FEM count info: \n"
              << ipc.abd_fem_count_info << std::endl;


    printf("vertNum: %d      tetraNum: %d      faceNum: %d\n",
           ipc.vertexNum,
           ipc.tetrahedraNum,
           ipc.surface_Num);
    printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum, ipc.edge_Num);
    printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n",
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    //ipc.USE_MAS = false;
    ipc.MALLOC_DEVICE_MEM();

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tetMesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id,
                d_tetMesh.collision_skip_matrix, d_tetMesh.collision_body_num);
    ipc._point_body_id = d_tetMesh.point_id_to_body_id;

    if(ipc.pcg_data.P_type && true)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                                    tetMesh.abd_vertexOffset,
                                                    neighborListSize,
                                                    ipc._collisonPairs,
                                                    tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));


#ifdef USE_QUADRATIC_BENDING
    // Precompute Q matrices for quadratic bending
    if(tetMesh.tri_edges.size() > 0)
    {
        printf("Precomputing Q matrices for quadratic bending (%zu edges)...\n",
               tetMesh.tri_edges.size());

        // Allocate host memory for Q matrices
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());

        // Download rest positions, edges, and adjacency from device
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));

        // Call precomputation function (defined in femEnergy.cu)
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),  // CPU端叫tri_edges_adj_points
                            tetMesh.tri_edges.size(),
                            Q_host.data());

        // Upload Q matrices to device
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));

        printf("Quadratic bending Q matrices uploaded successfully.\n");

        // Optional: Print first Q matrix for verification
        if(tetMesh.tri_edges.size() > 0)
        {
            printf("First Q matrix:\n");
            for(int i = 0; i < 4; i++)
            {
                printf("  [%10.6f %10.6f %10.6f %10.6f]\n",
                       Q_host[0](i, 0),
                       Q_host[0](i, 1),
                       Q_host[0](i, 2),
                       Q_host[0](i, 3));
            }

            // Check for NaN or Inf
            int nan_count = 0;
            int inf_count = 0;
            for(size_t i = 0; i < Q_host.size(); i++)
            {
                for(int r = 0; r < 4; r++)
                {
                    for(int c = 0; c < 4; c++)
                    {
                        if(std::isnan(Q_host[i](r, c)))
                            nan_count++;
                        if(std::isinf(Q_host[i](r, c)))
                            inf_count++;
                    }
                }
            }
            if(nan_count > 0 || inf_count > 0)
            {
                printf("WARNING: Q matrices contain %d NaN values and %d Inf values!\n",
                       nan_count,
                       inf_count);
            }
        }
    }
#endif


    ipc.buildBVH();
    ipc.setup_surface_mesh_bodies(tetMesh);
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer, linear_system_buff_scale);

    // Initialize joint constraints (revolute + fixed + prismatic) after ABD system is ready
    if(!tetMesh.joint_constraints.empty() || !tetMesh.prismatic_constraints.empty())
    {
        ipc.init_joint_constraints_from_mesh(tetMesh);
    }

    printf("bboxDiagSize2: %f\n", ipc.bboxDiagSize2);
    printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n",
           tetMesh.maxConer.x,
           tetMesh.maxConer.y,
           tetMesh.maxConer.z,
           tetMesh.minConer.x,
           tetMesh.minConer.y,
           tetMesh.minConer.z);

    printf("restSNKE: %f\n", ipc.RestNHEnergy);
    ipc.buildCP();

    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / motion_rate;
    //ipc.animation_fullRate = ipc.animation_subRate;
    ipc.computeXTilta(d_tetMesh, 1);
    ///////////////////////////////////////////////////////////////////////////////////

    ipc.create_LinearSystem(d_tetMesh);

    if(!ipc.m_skip_all_collision && ipc.edge_Num > 0)
    {
        bvs.resize(2 * ipc.edge_Num - 1);
        nodes.resize(2 * ipc.edge_Num - 1);
        CUDA_SAFE_CALL(cudaMemcpy(
            &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(
            &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
    }
}


void outputAnimationMeshInfo(string pathCloth, string pathBody)
{
    std::stringstream ss;
    ss << pathCloth;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    ofstream    outSurf(file_path);

    map<int, int> meshToSurf;
    for(int i = 0; i < bodyVertOffset; i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.triangles.size(); i++)
    {
        const auto& tri = tetMesh.triangles[i];
        outSurf << "f " << tri.x + 1 << " " << tri.y + 1 << " " << tri.z + 1 << endl;
    }
    outSurf.close();

    std::stringstream ss2;
    ss2 << pathBody;
    ss2.fill('0');
    ss2.width(5);
    ss2 << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss2 << ".obj";
    std::string file_path2 = ss2.str();
    ofstream    outSurf2(file_path2);

    //map<int, int> meshToSurf;
    for(int i = bodyVertOffset; i < tetMesh.vertexes.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf2 << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < clothFaceOffset; i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf2 << "f " << tri.x + 1 - bodyVertOffset << " " << tri.y + 1 - bodyVertOffset
                 << " " << tri.z + 1 - bodyVertOffset << endl;
    }
    outSurf2.close();
    surfNumId++;
}
bool pri = true;
static std::string g_output_path;
static bool        g_output_dirs_created = false;

void display(void)
{
    if(!g_headless_benchmark)
        draw_Scene3D();

    if(!g_output_dirs_created)
    {
        std::filesystem::exists(std::string{gipc::output_dir()})
            || std::filesystem::create_directory(std::string{gipc::output_dir()});
        g_output_path = std::string{gipc::output_dir()} + "saveSurface/";
        std::filesystem::exists(g_output_path) || std::filesystem::create_directory(g_output_path);
        g_output_dirs_created = true;
    }

    if(stop)
        return;

    auto frame_start = std::chrono::high_resolution_clock::now();

    if(g_settling_frames > 0)
    {
        ipc.m_abd_system->parms.max_revolute_step_per_frame  = 0.05;
        ipc.m_abd_system->parms.max_prismatic_step_per_frame = 0.01;
    }

    if(g_trajectory_playback_enabled)
        update_trajectory(ipc.IPC_dt);

    if(g_joint_control_enabled)
    {
        ipc.update_joint_angle_targets_from_mesh(tetMesh);
    }

    auto solver_start = std::chrono::high_resolution_clock::now();
    ipc.IPC_Solver(d_tetMesh);

    if(g_settling_frames > 0)
    {
        --g_settling_frames;
        if(g_settling_frames == 0)
        {
            ipc.m_abd_system->parms.max_revolute_step_per_frame  = 1;
            ipc.m_abd_system->parms.max_prismatic_step_per_frame = 0.002;
        }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto solver_end = std::chrono::high_resolution_clock::now();

    double solver_ms = std::chrono::duration<double, std::milli>(solver_end - solver_start).count();

    if(ipc.animation && !g_headless_benchmark)
    {
        std::string filename =
            "triMesh/body4/postcvpr_big_body_" + std::to_string(frameId + 1) + ".obj";
        frameId++;
        tetMesh.load_animation(filename, 1, make_double3(-1, -0.5, -0.5));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                  tetMesh.targetPos.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    }

    if(!ipc.m_skip_all_collision && ipc.edge_Num > 0)
    {
        CUDA_SAFE_CALL(cudaMemcpy(
            &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(
            &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
    }

    if(!g_skip_rendering || saveSurface)
    {
        CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                  ipc._vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
    }

    if(saveSurface)
    {
        saveSurfaceMesh(g_output_path);
    }

    if(screenshot && !g_headless_benchmark)
    {
        std::stringstream ss;
        ss << "saveScreen/step_";
        ss.fill('0');
        ss.width(5);
        ss << step / 1;
        std::string file_path = ss.str();
        SaveScreenShot(window_width, window_height, file_path);
    }
    step++;

    auto frame_end = std::chrono::high_resolution_clock::now();
    double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    printf("step: %d | solver: %.1f ms | frame_total: %.1f ms (%.1f FPS)\n",
           step, solver_ms, frame_ms, 1000.0 / frame_ms);

    if(!g_headless_benchmark)
    {
        char title[160];
        snprintf(title, sizeof(title),
                 "StiffGIPC wrecking-ball | step %d | solver %.1f ms | %.1f FPS",
                 step, solver_ms, 1000.0 / frame_ms);
        glutSetWindowTitle(title);
    }
}

void init(void)
{
    Init_CUDA();

    //main2();

    GLenum err = glewInit();
    if(GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    glClearColor(0.0, 0.0, 0.0, 1.0);


    LoadSettings();

    ipc.build_gipc_system(d_tetMesh);

    initScene();

    if(!isSetShader)
    {
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (float)window_width / window_height, 0.1f, 500.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    else
    {
        glGenBuffers(1, &PN_vbo_);
        glGenVertexArrays(1, &VAO);
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL2_Init();
    // Resize callback for ImGui
    ImGui_ImplGLUT_ReshapeFunc(static_cast<int>(window_width), static_cast<int>(window_height));
}


void idle_func()
{
    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
    ImGui_ImplGLUT_ReshapeFunc(width, height);

    glViewport(0, 0, width, height);
    if(!isSetShader)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        gluPerspective(45.0, (float)width / height, 0.1, 500.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
}

void keyboard_func(unsigned char key, int x, int y)
{
    // Forward to ImGui
    ImGui_ImplGLUT_KeyboardFunc(key, x, y);
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureKeyboard)
    {
        glutPostRedisplay();
        return;
    }

    if(key == 'w')
    {
        zTrans += .3f;
    }

    if(key == 's')
    {
        zTrans -= .3f;
    }

    if(key == 'a')
    {
        xTrans += .3f;
    }

    if(key == 'd')
    {
        xTrans -= .3f;
    }

    if(key == 'q')
    {
        yTrans -= .3f;
    }

    if(key == 'e')
    {
        yTrans += .3f;
    }

    if(key == '/')
    {
        screenshot = !screenshot;
    }

    if(key == '9')
    {
        saveSurface = !saveSurface;
    }

    if(key == 'k')
    {
        drawSurface = !drawSurface;
    }

    if(key == 'f')
    {
        drawbvh = !drawbvh;
    }

    if(key == ' ')
    {
        stop = !stop;
    }
    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
    ImGui_ImplGLUT_SpecialFunc(key, x, y);
    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    ImGui_ImplGLUT_MouseFunc(button, state, x, y);
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureMouse)
    {
        glutPostRedisplay();
        return;
    }

    if(state == GLUT_DOWN)
    {
        buttonState = 1;
    }
    else if(state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    ImGui_ImplGLUT_MotionFunc(x, y);
    ImGuiIO& io = ImGui::GetIO();
    if(io.WantCaptureMouse)
    {
        glutPostRedisplay();
        return;
    }

    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if(buttonState == 1)
    {
        xRot += dy / 5.0f;
        yRot += dx / 5.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}


void SpecialKey(GLint key, GLint x, GLint y)
{
    ImGui_ImplGLUT_SpecialFunc(key, x, y);

    if(key == GLUT_KEY_DOWN)
    {
        change = true;
        initPath -= 1;
        if(initPath < 0)
        {
            initPath = obj_pathes.size() - 1;
        }
    }

    if(key == GLUT_KEY_UP)
    {
        change = true;
        initPath += 1;
        if(initPath == obj_pathes.size())
        {
            initPath = 0;
        }
    }
    glutPostRedisplay();
}


int main(int argc, char** argv)
{
    bool verify = false;  // --verify: print per-step vertex checksum for correctness diff
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
        {
            g_scene_no = atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "--headless") == 0)
        {
            // Benchmark harness flag: run any scene headless and time it.
            // Does not touch collision/solver settings, so per-step solver
            // time reflects the scene's own configuration.
            g_headless_benchmark = true;
            g_skip_rendering     = true;
            if(i + 1 < argc && argv[i + 1][0] != '-')
                g_headless_max_steps = atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "--verify") == 0)
        {
            verify = true;
        }
        else if(argv[i][0] != '-')
        {
            g_scene_no = atoi(argv[i]);
        }
    }
    printf(">>> scene_no = %d\n", g_scene_no);

    glutInit(&argc, argv);

    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("FEM");

    init();

    if(g_headless_benchmark)
    {
        glutHideWindow();
        stop = false;
        std::cout << "[headless] Entering headless benchmark loop..." << std::endl;

        int max_steps = g_headless_max_steps;
        auto total_start = std::chrono::high_resolution_clock::now();

        for(int s = 0; s < max_steps; s++)
        {
            display();
            if(verify)
            {
                CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                          ipc._vertexes,
                                          ipc.vertexNum * sizeof(double3),
                                          cudaMemcpyDeviceToHost));
                long double s1 = 0.0L, s2 = 0.0L;
                for(int v = 0; v < ipc.vertexNum; v++)
                {
                    const double3& q = tetMesh.vertexes[v];
                    s1 += (long double)q.x + q.y + q.z;
                    s2 += (long double)q.x * q.x + (long double)q.y * q.y
                          + (long double)q.z * q.z;
                }
                printf("VERIFY step %d  sum=%.12Le  sumsq=%.12Le\n",
                       s, s1, s2);
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "==============================" << std::endl;
        std::cout << "[headless] Completed " << max_steps << " steps in "
                  << total_s << " s (" << (max_steps / total_s) << " FPS)" << std::endl;
        std::cout << "==============================" << std::endl;
        return 0;
    }

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_MULTISAMPLE);
    glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(&SpecialKey);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutIdleFunc(idle_func);

    glutMainLoop();
}
