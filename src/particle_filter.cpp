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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 100;
	
	float std_x =     std[0];
	float std_y =     std[1];
	float std_theta = std[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	for(int i = 0 ; i < num_particles ; i++){
		Particle particle;
		particle.x     = dist_x(gen);
		particle.y	= dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		
	}

	weights.resize(num_particles);
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	default_random_engine gen;
	float std_x =     std_pos[0];
	float std_y = 	  std_pos[1];
	float std_theta = std_pos[2];
	
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for(int i =0 ; i < num_particles; i++){
		 if (fabs(yaw_rate) < 0.00001) {  
      			particles[i].x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
      			particles[i].y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
    		} 
    		else {
      			particles[i].x =     particles[i].x + (velocity / yaw_rate * (sin(particles[i].theta + 						     yaw_rate*delta_t) - sin(particles[i].theta)));
      			particles[i].y =     particles[i].y + (velocity / yaw_rate * (cos(particles[i].theta) - 						     cos(particles[i].theta + yaw_rate*delta_t)));
      			particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
    		}
	
		 particles[i].x     = particles[i].x + dist_x(gen);
    		 particles[i].y     = particles[i].y + dist_y(gen);
    		 particles[i].theta = particles[i].theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (unsigned int i = 0; i < observations.size(); i++) {
		 double minDistance =  numeric_limits<double>::max();
		  int tempID = -1;
		  for (unsigned int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y,  predicted[j].x,  predicted[j].y);
			
			if (distance  <  minDistance ) {
       				 minDistance = distance;
				 tempID =  predicted[j].id;
      			}
		  }
		  observations[i].id = tempID;	
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	for(int p = 0 ; p < num_particles; p++){
		double p_x     = particles[p].x;
    		double p_y     = particles[p].y;
   		double p_theta = particles[p].theta;

		vector<LandmarkObs> transformed_obs;
		
		for(int o = 0 ; o < observations.size(); o++){
			  float obs_x = observations[o].x;
      			  float obs_y = observations[o].y;
      			  int  obs_id = observations[o].id;
			  
			  double t_x = cos(p_theta) * obs_x - sin(p_theta) * obs_y + p_x;
      			  double t_y = sin(p_theta) * obs_x + cos(p_theta) * obs_y + p_y;
      			  transformed_obs.push_back(LandmarkObs{obs_id , t_x, t_y});
		}


		 vector<LandmarkObs> predictions;
		 for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); l++) {
			  	
			    float lm_x = map_landmarks.landmark_list[l].x_f;
      			   float lm_y = map_landmarks.landmark_list[l].y_f;
      			  int lm_id = map_landmarks.landmark_list[l].id_i;	
		          if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

        		predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      			}
      		}
    		

	dataAssociation(predictions, transformed_obs);
	particles[p].weight = 1.0;
	
	
	for(int i = 0 ; i < transformed_obs.size(); i++){
		int j = 0;
		for( ; j < predictions.size(); j++){
			if(predictions[j].id == transformed_obs[i].id)break;
		}
		double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(predictions[j].x-transformed_obs[i].x,2)/(2*pow(std_landmark[0], 2)) + (pow(predictions[j].y-transformed_obs[i].y,2)/(2*pow(std_landmark[1], 2))) ) );
	
		particles[p].weight *= weight;
	}
		weights[p] = particles[p].weight;
		
   }
	
	
}


void ParticleFilter::resample() {
 
 vector<Particle> new_particles;
  default_random_engine gen;

  uniform_int_distribution<int> indexDist(0, num_particles-1);
  auto index = indexDist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> weightDist(0.0, max_weight);

  double beta = 0.0;

  for (int i = 0; i < num_particles; i++) {
    beta += weightDist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
