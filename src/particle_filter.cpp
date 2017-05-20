/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// initialize number of particles
	num_particles = 100;

	// initialize normal distributions around GPS measurements
	normal_distribution<double> x_dist(x, std[0]);
	normal_distribution<double> y_dist(y, std[1]);
	normal_distribution<double> theta_dist(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		// create a particle, assigning it a random initial x, y, and theta coordinates
		Particle p;
		p.id = i;
		p.x = x_dist(gen);
		p.y = y_dist(gen);
		p.theta = theta_dist(gen);

		// initialize particle's weight as 1.0
		p.weight = 1.0;

		// add particle to particles vector
		particles.push_back(p);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// initialize normal distributions for x, y, and theta error (with mean = 0)
	normal_distribution<double> x_dist(0, std_pos[0]);
	normal_distribution<double> y_dist(0, std_pos[1]);
	normal_distribution<double> theta_dist(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle &p = particles[i];

		if (fabs(yaw_rate) > 0.00001)
		{
			// if yaw_rate != 0, predict new x, y, and theta using constant turn rate motion model
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity / yaw_rate * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta));
			p.theta += yaw_rate * delta_t;
		}
		else
		{
			// if yaw_rate == 0, predict new x, y, and theta using linear motion model
			p.x += velocity * cos(p.theta) * delta_t;
			p.y += velocity * sin(p.theta) * delta_t;
		}

		// add random noise to predicted positions
		p.x += x_dist(gen);
		p.y += y_dist(gen);
		p.theta += theta_dist(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i)
	{
		LandmarkObs &obs = observations[i];

		// initialize minimum distance to be very high
		double min_distance = 1000;

		for (int j = 0; j < predicted.size(); ++j)
		{
			LandmarkObs pred = predicted[j];

			// calculate distance between landmark and observation
			double distance = dist(pred.x, pred.y, obs.x, obs.y);

			if (distance < min_distance)
			{
				// if distance is lower than current minimum,
				// make min_distance equal to the current distance
				// and make the observation's id number equal
				// to the landmark
				min_distance = distance;
				obs.id = pred.id;
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

	// clear weights vector
	weights.clear();

	for (int i = 0; i < num_particles; ++i)
	{
		Particle &p = particles[i];

		// initialize vector for observations transformed to map coordinates
		vector<LandmarkObs> observations_transformed;
		LandmarkObs obs_trans;

		for (int j = 0; j < observations.size(); ++j)
		{
			LandmarkObs obs = observations[j];

			// transform observation to map coordinates
			obs_trans.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
			obs_trans.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);

			// add transformed observation to vector
			observations_transformed.push_back(obs_trans);
		}

		// initialize vector for landmarks within sensor range
		vector<LandmarkObs> predicted;
		LandmarkObs pred;
		predicted.clear();

		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
			// store map landmark info in LandmarkObs struct
			pred.x = map_landmarks.landmark_list[j].x_f;
			pred.y = map_landmarks.landmark_list[j].y_f;
			pred.id = map_landmarks.landmark_list[j].id_i;

			// check distance between particle and landmark
			double distance = dist(p.x, p.y, pred.x, pred.y);

			if (distance < sensor_range)
			{
				// if landmark is within sensor range, then add it
				// to the list of predicted landmarks
				predicted.push_back(pred);
			}
		}

		// perform data association to identify landmarks for each observation
		dataAssociation(predicted, observations_transformed);

		// initialize particle weight as 1.0 since it will be
		// updated as a product
		p.weight = 1.0;

		// initialize parameters for bivariate Gaussian distribution
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];
		double sigma_x2 = pow(sigma_x, 2);
		double sigma_y2 = pow(sigma_y, 2);

		for (int j = 0; j < observations_transformed.size(); ++j)
		{
			obs_trans = observations_transformed[j];

			for (int k = 0; k < predicted.size(); ++k)
			{
				pred = predicted[k];

				if (obs_trans.id == pred.id)
				{
					// find prediction that has the same id as each observation,
					// and calculate the square distance in x and y
					double x_diff2 = pow(obs_trans.x - pred.x, 2);
					double y_diff2 = pow(obs_trans.y - pred.y, 2);

					// update the particle weight using a bivariate normal distribution
					p.weight *= 1.0 / (2.0 * M_PI * sigma_x * sigma_y) * exp(-0.5 * (x_diff2 / sigma_x2 + y_diff2 / sigma_x2));

					break;
				}
			}
		}

		// add updated particle weight to weights vector
		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// initialize discrete distribution using weights vector
	discrete_distribution<int> p_dist(weights.begin(), weights.end());

	// initialize dummy particles vector
	vector<Particle> particles2;

	for (int i = 0; i < num_particles; ++i)
	{
		// choose random index based on weighted discrete distribution
		int rand_index = p_dist(gen);

		// find particle at chosen index
		Particle p = particles[rand_index];

		// add that particle to dummy particles vector
		particles2.push_back(p);
	}

	// once dummy particles vector is built,
	// copy it into the particles vector
	particles = particles2;
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
