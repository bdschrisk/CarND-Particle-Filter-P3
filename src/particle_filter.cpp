/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include "particle_filter.h"

using namespace std;

static default_random_engine generator(4211692);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	this->num_particles = 180;
	
	// create guassian noise
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;

		particle.id = i;
		particle.x = dist_x(generator);
		particle.y = dist_y(generator);
		particle.theta = dist_theta(generator);
		// Initialise weight to 1
		particle.weight = 1.0;

		this->weights.push_back(particle.weight);
		this->particles.push_back(particle);
	}

	this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_psi(0, std_pos[2]);

	for (int i = 0; i < this->num_particles; i++) {
		double x = this->particles[i].x;
		double y = this->particles[i].y;
		double theta = this->particles[i].theta;

		if (fabs(yaw_rate) > FLT_EPSILON) {
			x = x + (velocity / yaw_rate) * (sin(theta + (yaw_rate * delta_t)) - sin(theta));
			y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + (yaw_rate * delta_t)));
		}
		else {
			x = x + (velocity * delta_t) * sin(theta);
			y = y + (velocity * delta_t) * cos(theta);
		}

		theta = theta + (yaw_rate * delta_t);

		this->particles[i].x = x + dist_x(generator);
		this->particles[i].y = y + dist_y(generator);
		this->particles[i].theta = theta + dist_psi(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
	
	vector<LandmarkObs> current;

	unsigned int prediction_size = predicted.size();
	unsigned int obs_size = observations.size();

	for (unsigned int i = 0; i < prediction_size; i++) {
		
		double min_val = FLT_MAX;
		int min_idx = 0;
		
		for (unsigned int j = 0; j < obs_size; j++) {
			double distance = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
			if (distance < min_val) {
				min_val = distance;
				min_idx = j;
			}
		}

		LandmarkObs obs = observations[min_idx];
		current.push_back(obs);
	}

	observations = current;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	
	// define gaussian probability estimate //
	auto gaussianProbability = [] (double x, double xu, double y, double yu, double std_x, double std_y) {
		double x_exp = (pow(x - xu, 2) / (2.0 * pow(std_x, 2)));
		double y_exp = (pow(y - yu, 2) / (2.0 * pow(std_y, 2)));

		return (1.0 / (2.0 * M_PI * std_x*std_y)) * exp(-(x_exp + y_exp));
	};
	// define rotation and translation //
	auto translatePosition = [] (LandmarkObs obs, Particle p) {
		LandmarkObs pObs;

		pObs.id = obs.id;
		pObs.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
		pObs.y = p.y + (obs.x * sin(p.theta) + obs.y * cos(p.theta));

		return pObs;
	};

	unsigned int obs_size = observations.size();
	unsigned int landmark_size = map_landmarks.landmark_list.size();

	vector<LandmarkObs> all_landmarks;
	for (unsigned int l = 0; l < landmark_size; l++) {
		LandmarkObs obs;
		obs.id = map_landmarks.landmark_list[l].id_i;
		obs.x = map_landmarks.landmark_list[l].x_f;
		obs.y = map_landmarks.landmark_list[l].y_f;

		all_landmarks.push_back(obs);
	}

	for (int i = 0; i < this->num_particles; i++) {
		// convert observations to local particle coordinate
		Particle particle = this->particles[i];
		
		vector<LandmarkObs> predictions;

		for (unsigned int o = 0; o < obs_size; o++) {
			LandmarkObs obs = observations[o];
			LandmarkObs pObs;

			pObs = translatePosition(obs, particle);

			double distance = dist(particle.x, particle.y, pObs.x, pObs.y);
			if (distance < sensor_range) {
				predictions.push_back(pObs);
			}
		}
		
		vector<LandmarkObs> currentLandmarks;
		currentLandmarks = all_landmarks;
		// associate predictions with nearest neighbours
		dataAssociation(predictions, currentLandmarks);

		unsigned int pred_size = predictions.size();

		// avoid loss
		this->particles[i].weight = 1.0;

		// update weight for i-th particle using gaussian estimate
		for (unsigned int k = 0; k < pred_size; k++) {
			LandmarkObs pObs = predictions[k];
			LandmarkObs obs = currentLandmarks[k];

			// predicted map pos + associated landmark pos
			double prob = gaussianProbability(pObs.x, obs.x, pObs.y, obs.y, std_landmark[0], std_landmark[1]);
			this->particles[i].weight *= prob + DBL_EPSILON;
		}

		// update weights
		this->weights[i] = this->particles[i].weight;
	}

}

void ParticleFilter::resample() {

	discrete_distribution<int> particle_distribution(this->weights.begin(), this->weights.end());

	vector<Particle> new_particles;
	for (int i = 0; i < this->num_particles; i++) {
		Particle particle = this->particles[particle_distribution(generator)];
		new_particles.push_back(particle);
	}
	// update weights
	for (int i = 0; i < this->num_particles; i++) {
		this->weights[i] = new_particles[i].weight;
	}

	this->particles = new_particles;
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
