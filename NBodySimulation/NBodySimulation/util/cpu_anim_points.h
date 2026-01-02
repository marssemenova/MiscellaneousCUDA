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
#include "../include/glm/glm.hpp"
#include "../include/glm/gtc/matrix_transform.hpp"
#include <iostream>

struct CPUAnimPoints {
    float *points;
    float *a; // TODO: temp
    int init; // TODO: temp
    void* dataBlock;
    double range;
    void (*fAnim)(void*,int);
    void (*animExit)(void*);
    int n;

    /**
     * CPUAnimPoints constructor.
     * 
     * @param num - N.
     * @param r - Range of data, used to set eye coordinates.
     * @param initPos - Array (float4) of generated positions.
     * @param d - DataBlock object with program data.
     */
    CPUAnimPoints( int num, double r, float4* initPos, void* d = NULL) {
        n = num;
        range = r;
        dataBlock = d;
        points = (float*)malloc(n * sizeof(float) * 3);
        copyPoints(points, initPos);
        a = (float*)malloc(n * sizeof(float) * 3); // TODO: temp
        init = 0; // TODO: temp
    }

    /**
     * Method for copying points.
     * 
     * @param points - Pointer to a CPUAnimPoints object's points array.
     * @para pos - A float4 array from which to extract new point data.
     */
    void copyPoints(float *points, float4 *pos) {
        for (int x = 0; x < n; x++) {
            points[3 * x] = pos[x].x;
            points[3 * x + 1] = pos[x].y;
            points[3 * x + 2] = pos[x].z;
        }
    }

    void copyA(float* acc, float4* acceleration, int* inited) { // TODO: temp
        for (int x = 0; x < n; x++) {
            acc[3 * x] = acceleration[x].x;
            acc[3 * x + 1] = acceleration[x].y;
            acc[3 * x + 2] = acceleration[x].z;
            *inited = 1;
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
        glutInit(&c, &dummy);
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
        glutInitWindowSize( 1280, 720 );
        glutCreateWindow("nbody");
        glutDisplayFunc(my_display);
        glutIdleFunc(my_idle);
        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, ((float) 1280.0) / ((float) 720.0), 0.1, 100);
        glMatrixMode(GL_MODELVIEW);
        glutMainLoop();
    }

    static CPUAnimPoints** get_ptr(void) {
        static CPUAnimPoints* gPtsAnim;
        return &gPtsAnim;
    }

    // static method used for glut callbacks
    static void my_idle( void ) {
        static int ticks = 1;
        CPUAnimPoints* ptsAnimator = *(get_ptr());
        ptsAnimator->fAnim(ptsAnimator->dataBlock, ticks++ );
        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void my_display( void ) {
        CPUAnimPoints *ptsAnimator = *(get_ptr());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // camera
        glLoadIdentity();
        gluLookAt(ptsAnimator->range, ptsAnimator->range, ptsAnimator->range,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0);

        // draw bodies
        glPointSize(5.0f);
        glBegin(GL_POINTS);
            for (int x = 0; x < ptsAnimator->n; x++) {
                glColor4f(1.0, 0.0, 0.0, 1.0);
                glVertex3f(ptsAnimator->points[3 * x], ptsAnimator->points[3 * x + 1], ptsAnimator->points[3 * x + 2]);
                if (ptsAnimator->init) {
                    glColor4f(0.0, 1.0, 0.0, 1.0);
                    glVertex3f(ptsAnimator->points[3 * x] + ptsAnimator->a[3 * x], ptsAnimator->points[3 * x + 1] + ptsAnimator->a[3 * x + 1], ptsAnimator->points[3 * x + 2] + ptsAnimator->a[3 * x + 2]);
                }
            }
            
        glEnd();

        glutSwapBuffers();
    }
};
#endif  // __CPU_ANIM_POINTS_H__

