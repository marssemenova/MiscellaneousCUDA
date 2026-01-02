/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_ANIM_POINTS_H__
#define __CPU_ANIM_POINTS_H__

#include "gl_helper.h"
#include "vector_types.h"
#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include <iostream>
#include "shader.h"

struct CPUAnimPoints {
    float *points;
    void* dataBlock;
    double range;
    void (*fAnim)(void*,int);
    void (*animExit)(void*);
    int n;
    glm::mat4 Projection;
    glm::mat4 V;

    CPUAnimPoints( int num, double r, float4* initPos, void* d = NULL) {
        n = num;
        range = r;
        dataBlock = d;
        points = (float*)malloc(n * sizeof(float) * 3);
        copyPoints(points, initPos);
    }

    void copyPoints(float *points, float4 *pos) {
        for (int x = 0; x < n; x++) {
            points[3 * x] = pos[x].x;
            points[3 * x + 1] = pos[x].y;
            points[3 * x + 1] = pos[x].z;
        }
    }

    void anim_and_exit( void (*f)(void*,int), void(*e)(void*) ) {
        CPUAnimPoints** pointsAnimator = get_ptr();
        *pointsAnimator = this;
        fAnim = f;
        animExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        Projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.001f, 1000.0f);
        glm::vec3 eye = { (*pointsAnimator)->range, (*pointsAnimator)->range, (*pointsAnimator)->range };
        glm::mat4 V = glm::lookAt(eye, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( 1280, 720 );
        glutCreateWindow( "nbody" );
        glutDisplayFunc(Draw);
        glutIdleFunc( idle_func );
        glutMainLoop();
    }

    static CPUAnimPoints** get_ptr(void) {
        static CPUAnimPoints* gPtsAnim;
        return &gPtsAnim;
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        static int ticks = 1;
        CPUAnimPoints* ptsAnimator = *(get_ptr());
        ptsAnimator->fAnim(ptsAnimator->dataBlock, ticks++ );
        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        CPUAnimPoints*   ptsAnimator = *(get_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        //glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glutSwapBuffers();
    }
};


#endif  // __CPU_ANIM_POINTS_H__

