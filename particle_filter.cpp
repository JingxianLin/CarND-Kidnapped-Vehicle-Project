/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Jingxian Lin
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;
    weights.resize(num_particles);
    particles.resize(num_particles);

    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);
    for (int i = 0; i < num_particles; ++i) {
    	particles[i].id = i + 1;
    	
    	particles[i].x = dist_x(gen);
    	particles[i].y = dist_y(gen);
    	particles[i].theta = dist_theta(gen);

    	particles[i].weight = 1.0;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;
    std::normal_distribution<double> noise_x(0.0, std_pos[0]);
    std::normal_distribution<double> noise_y(0.0, std_pos[1]);
    std::normal_distribution<double> noise_theta(0.0, std_pos[2]);
    yaw_rate = (fabs(yaw_rate) < 0.0001) ? 0.0001 : yaw_rate;

    for (int i = 0; i < num_particles; ++i) {
        particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + noise_x(gen);
        particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + noise_y(gen);
    	particles[i].theta += yaw_rate * delta_t + noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); ++i) {
        double min_dist = 50;

        for (int j = 0; j < predicted.size(); ++j) {
        	double dist_to_obs = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        	if (dist_to_obs <= min_dist) {
                min_dist = dist_to_obs;
                observations[i].id = predicted[j].id;
        	}
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	double tot = 0;
    for (int i = 0; i < num_particles; ++i) {
    	double xp = particles[i].x;
    	double yp = particles[i].y;
    	double tp = particles[i].theta;
    	std::vector<LandmarkObs> predicted;
       	long double p = 1.0;
    	for (int k = 0; k < observations.size(); ++k) {
    		double xtr = xp + observations[k].x * cos(tp) - observations[k].y * sin(tp);
    		double ytr = yp + observations[k].x * sin(tp) + observations[k].y * cos(tp);

            double xm, ym;
            double min_dist = sensor_range;
            for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
                double xlm = map_landmarks.landmark_list[j].x_f;
                double ylm = map_landmarks.landmark_list[j].y_f;
                double dist_to_lm = dist(xp, yp, xlm, ylm);

                if (dist_to_lm <= sensor_range) {
                	double dist_to_obs = dist(xtr, ytr, xlm, ylm);
                	if (dist_to_obs <= min_dist) {
                		min_dist = dist_to_obs;
                        xm = xlm;
                        ym = ylm;;
                	}
                }
            }
            double xd2 = (xtr - xm) * (xtr - xm);
            double yd2 = (ytr - ym) * (ytr - ym);
            p *= exp(-xd2 / 2 / std_landmark[0] / std_landmark[0] - yd2 / 2 / std_landmark[1] / std_landmark[1]) / 2 / M_PI / std_landmark[0] / std_landmark[1];
    	}

        particles[i].weight = p;
        weights[i] = p;
        tot += p;
    }
    for (int l = 0; l < weights.size(); ++l) {
    	weights[l] /= tot;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    std::vector<Particle> temp_part;
    for (int i = 0; i < num_particles; ++i) {
    	int weighted_index = distribution(gen);
    	temp_part.push_back(particles[weighted_index]);
    }
    particles = temp_part;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
