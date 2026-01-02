#include "gl_helper.h"
#include "../include/glm/glm.hpp"
#include "../include/glad/include/glad/glad.h"
#include "shader.h"

class Point {

private:
	GLuint VertexArrayID;
	GLuint programID;
	GLuint MVPID;
	GLuint FlareSizeID;
	GLuint colorID;

	GLuint vertexbuffer;

	GLuint pointTex;
	GLuint pointTexID = 0;
	int textureSize = 32;

	void setupVAO(glm::vec3 pos) {
		glGenVertexArrays(1, &VertexArrayID);
		glBindVertexArray(VertexArrayID);

		programID = LoadShaders( "PointShader.vertexshader", "PointShader.geometryshader", "PointShader.fragmentshader" );

		glUseProgram(programID);

		MVPID = glGetUniformLocation(programID, "MVP");
		colorID = glGetUniformLocation(programID, "modelcolor");

		FlareSizeID = glGetUniformLocation(programID, "flare_size");
		GLfloat flareSize[2] = { ((float) textureSize) / (1.0f*SCR_WIDTH), ((float) textureSize) / (1.0f*SCR_HEIGHT)};

		glUniform2f(FlareSizeID, flareSize[0], flareSize[1]);

	 	GLuint TextureID  = glGetUniformLocation(programID, "tex");
		glUniform1i(TextureID, pointTexID);
		glUseProgram(0);

		glGenBuffers(1, &vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		std::vector<float> pt = {pos.x, pos.y, pos.z};
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3, pt.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,                  // position attrib
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		// point texture now
		glGenTextures(1, &pointTex);
		glBindTexture(GL_TEXTURE_2D, pointTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureSize, textureSize, 0, GL_RED, GL_FLOAT, 0);
		// glCreateTextures(GL_TEXTURE_2D, 1, &pointTex);
		// glTextureStorage2D(pointTex, 1, GL_R32F, textureSize, textureSize);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLfloat texturePixels[textureSize*textureSize];
		//generate a discrete 2D Gaussian

		float sigma2 = textureSize * 0.5f;
		float A = 1.0f;
		for (int i = 0; i < textureSize; ++i) {
			float i1 = i-textureSize / 2.0f;
			for (int j = 0; j < textureSize; ++j) {
				float j1 = j - textureSize / 2.0f;
				texturePixels[i*textureSize + j] =
					pow(A*exp(-1.0f*((i1*i1)/(2*sigma2) + (j1*j1)/(2*sigma2))), 2.2);
			}
		}

		//fill server side texture data
		glTexSubImage2D(
			GL_TEXTURE_2D,		//target tex
			0, 					//level
			0,					//xoffset
			0,					//yoffset
			textureSize,		//width
			textureSize,		//height
			GL_RED, 			//texture format
			GL_FLOAT, 			//pixel data type
			texturePixels		//the data
		);

		glBindTexture(GL_TEXTURE_2D, 0);
	}


public:
	Point(glm::vec3 pos) {
		setupVAO(pos);
	}

	void draw(glm::mat4 M, glm::mat4 V, glm::mat4 P, glm::vec4 color) {
		glm::mat4 MVP = P*V*M;

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		glBindVertexArray(VertexArrayID);

		glUseProgram(programID);
		glUniformMatrix4fv(MVPID, 1, GL_FALSE, &MVP[0][0]);
		glUniform4fv(colorID, 1, &color[0]);

		//glActiveTexture(GL_TEXTURE0);
		//glEnable(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, pointTex);

		glDrawArrays(GL_POINTS, 0, 1);

		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);
		glBindVertexArray(0);
		glDisable(GL_BLEND);

	}

};
