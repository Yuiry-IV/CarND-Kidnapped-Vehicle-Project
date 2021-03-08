/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cassert>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // Set the number of particles
  
  // create normal distributions
  std::default_random_engine gen;
  std::normal_distribution<double> normal_distribution_x_init(x, std[0]);
  std::normal_distribution<double> normal_distribution_y_init(y, std[1]);
  std::normal_distribution<double> normal_distribution_theta_init(theta, std[2]);
  
  particles.clear();
  weights.clear();
  particles.reserve(num_particles);
  weights.reserve(num_particles);
  const double initial_weight=1.0;
  
  for (unsigned int i=0; i<num_particles; ++i){
      const Particle new_particle{
         int(i),
         normal_distribution_x_init(gen),
         normal_distribution_y_init(gen),
         normal_distribution_theta_init(gen),
         initial_weight };
      
      particles.push_back(new_particle);
      weights.push_back(initial_weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   
  // create normal distributions for process noise
  std::default_random_engine gen;
  
  std::normal_distribution<double> normal_distribution_x(0.0, std_pos[0]);
  std::normal_distribution<double> normal_distribution_y(0.0, std_pos[1]);
  std::normal_distribution<double> normal_distribution_theta(0.0, std_pos[2]);
  
  for (unsigned int i=0; i<num_particles; i++) {
    auto &p=(particles[i]);
    if (std::fabs(yaw_rate) > 1e-3) {
      const double p_theta = p.theta + delta_t * yaw_rate;
      p.x += (velocity / yaw_rate) * (std::sin(p_theta)- std::sin(p.theta));
      p.y += (velocity / yaw_rate) * (std::cos(p.theta) - std::cos(p_theta));
      p.theta = p_theta;    
    } else {
      p.x += velocity * std::cos(p.theta) * delta_t;
      p.y += velocity * std::sin(p.theta) * delta_t;
    }
    
    // add random noise
    p.x += normal_distribution_x(gen);
    p.y += normal_distribution_y(gen);
    p.theta += normal_distribution_theta(gen);
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weights_sum = 0.0;
  
  for (unsigned int i=0; i<num_particles; ++i){      
    // convert observed landmarks into world space from POV of particle
    std::vector<LandmarkObs> observations_particle_worldspace;
    observations_particle_worldspace.reserve(observations.size());
    
    const double cos_particle_theta = std::cos(particles[i].theta);
    const double sin_particle_theta = std::sin(particles[i].theta);
    
    for (unsigned int index_obs=0; index_obs<observations.size(); ++index_obs){
      auto const & p(particles[i]);
      auto const & obs(observations[index_obs]);
      
      const LandmarkObs landmark{ -1,
         p.x + obs.x*cos_particle_theta - obs.y*sin_particle_theta,
         p.y + obs.x*sin_particle_theta + obs.y*cos_particle_theta};
         
      observations_particle_worldspace.push_back(landmark);
    }
    
    // associate known map landmark to observed landmarks
    // nearest-neighbor association
    double weight = 1.0;
    
    for (unsigned int j=0; j<observations_particle_worldspace.size(); ++j) {
      double min_distance = sensor_range;
      const Map::single_landmark_s* closest_lm_ptr = nullptr;
      const double obs_x = observations_particle_worldspace[j].x;
      const double obs_y = observations_particle_worldspace[j].y;
      
      for (unsigned int k=0; k<map_landmarks.landmark_list.size(); ++k) {
        const double lm_x = map_landmarks.landmark_list[k].x_f;
        const double lm_y = map_landmarks.landmark_list[k].y_f;
        const double distance = dist(obs_x, obs_y, lm_x,  lm_y);
        if (distance < min_distance) {
          min_distance = distance;
          closest_lm_ptr = &map_landmarks.landmark_list[k];
        }
      }

      if (closest_lm_ptr) {
        // Multi-Variate Gaussian
        const double mvg_part1 = 1/(2.0*M_PI*std_landmark[0]*std_landmark[1]);
        const double x_dist    = obs_x - closest_lm_ptr->x_f;
        const double y_dist    = obs_y - closest_lm_ptr->y_f;
        const double mvg_part2 = ((x_dist*x_dist) / (2*std_landmark[0]*std_landmark[0])) + 
                                 ((y_dist*y_dist) / (2*std_landmark[1]*std_landmark[1]));
        const double mvg       = mvg_part1 * std::exp(-mvg_part2);
        weight *= mvg;
      }
    }   
    weights[i] = weight;
    particles[i].weight = weight;
    weights_sum += weight;
  }
    
  // normalize weights
  for (unsigned int i=0; i<num_particles; ++i){
      particles[i].weight /= weights_sum;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<Particle> resampled_particles;
    resampled_particles.reserve(particles.size());
    for (unsigned int i=0; i<particles.size(); ++i){
        const double rand_num = dis(gen);
        double particle_weights = 0.0;
        unsigned int j = 0;
        for ( ; particle_weights < rand_num; ++j){
            assert( j<particles.size() );
            particle_weights += particles[ j ].weight;
        }
        resampled_particles.push_back(particles[j-1]);
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
