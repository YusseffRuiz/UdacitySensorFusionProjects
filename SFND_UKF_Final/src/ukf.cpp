#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2

  //Velocity 25m/s, max_acc = 2.5, half = 1.15
  std_a_ = 1.25*1.25;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    is_initialized_ = false;

    // = MatrixXd(n_x_, 2*n_aug_+1);
    weights_ = VectorXd(2*n_aug_+1);
    //Filling Weight vector
    weights_.fill(1/(2*(n_aug_+lambda_)));
    weights_(0) = lambda_/(lambda_+n_aug_);

    nis_lidar_ = 0.0;
    nis_radar_ = 0.0;


}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // Read lidar and Radar data

      if(!is_initialized_){
          if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {  // laser measurement
              // read measurements, Laser only has values for X and Y
              x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0 ,0, 0; //x  = [px py v psi psiDot]'
              //P_.setIdentity();
              P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
                    0, std_laspy_*std_laspy_, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 1; //Covariance Matrix for laser



          } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
              // Skip Radar measurements ro, phi, ro_dot
              double ro = meas_package.raw_measurements_(0);
              double phi = meas_package.raw_measurements_(1);
              double ro_dot = meas_package.raw_measurements_(2);

              double x = ro * sin(phi);
              double y = ro * cos(phi);
              double v = ro_dot;
              x_ << x, y, v, phi, ro_dot;
              //P_.setIdentity();
              P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
                      0, std_radphi_ * std_radphi_, 0, 0, 0,
                      0, 0, std_radrd_ * std_radrd_, 0, 0,
                      0, 0, 0, 1, 0,
                      0, 0, 0, 0, 1;
          }
          else{
              exit(-1);
          }
          time_us_ = meas_package.timestamp_;
          is_initialized_ = true;
      }

    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    // Predict the next states and covariance matrix
    Prediction(dt);

    // Update the next states and covariance matrix
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
        UpdateLidar(meas_package);
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
        UpdateRadar(meas_package);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    VectorXd x_aug = VectorXd (n_aug_);
    MatrixXd p_aug = MatrixXd (n_aug_, n_aug_);
    MatrixXd  xSig_aug = MatrixXd (n_aug_, 2*n_aug_+1);
    x_aug.head(5) = x_;
    x_aug(5) = 0.0;
    x_aug(6) = 0.0;

    p_aug.fill(0.0);
    p_aug.topLeftCorner(5, 5) = P_;
    p_aug(n_x_,n_x_) = std_a_*std_a_;
    p_aug(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;



    // Create square root matrix
    MatrixXd L = p_aug.llt().matrixL();
    //Create augmented Sigma Points
    xSig_aug.col(0) = x_aug;
    for (int i = 0; i< n_aug_; ++i) {
        xSig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        xSig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }


    // Predict Sigma Points
    Xsig_pred_ = MatrixXd (n_x_, 2*n_aug_+1);
    for (int i = 0; i< 2*n_aug_+1; ++i) {
        // extract values for better readability
        double p_x = xSig_aug(0,i);
        double p_y = xSig_aug(1,i);
        double v = xSig_aug(2,i);
        double yaw = xSig_aug(3,i);
        double yawd = xSig_aug(4,i);
        double nu_a = xSig_aug(5,i);
        double nu_yawdd = xSig_aug(6,i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        // write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
    //Predict State
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }
    //Predict Covariance Matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    int n_z = 2; // measurement Dimension
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    // transform sigma points into measurement space
    for(int i = 0; i< (2*n_aug_+1); i++){
        float px = Xsig_pred_(0,i);
        float py = Xsig_pred_(1,i);


        Zsig(0,i) = px;
        Zsig(1,i) = py;
    }

    // calculate mean predicted measurement
    z_pred.fill(0.0);
    for(int i = 0; i< (2*n_aug_+1); i++){
        z_pred = z_pred + weights_(i)*Zsig.col(i);
    }

    // calculate innovation covariance matrix S

    S.fill(0.0);
    for(int i = 0; i< 2*n_aug_+1; i++){
        VectorXd zDiff = Zsig.col(i) - z_pred;
        while (zDiff(1)> M_PI) zDiff(1)-=2.*M_PI;
        while (zDiff(1)<-M_PI) zDiff(1)+=2.*M_PI;
        S = S + weights_(i)*zDiff*zDiff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    S = S + R;


    // UPDATE
    // Cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd  z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    //NIS calculation
    nis_lidar_ = z_diff.transpose()*S.inverse()*z_diff;



}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    int n_z = 3; // measurement Dimension
    // create matrix for cross correlation Tc
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    // transform sigma points into measurement space
    for(int i = 0; i< (2*n_aug_+1); i++){
        float px = Xsig_pred_(0,i);
        float py = Xsig_pred_(1,i);
        float v = Xsig_pred_(2,i);
        float yaw = Xsig_pred_(3,i);

        float p = sqrt(px*px + py*py);
        float delta = atan2(py,px);
        float p_dot = (px*cos(yaw)*v+py*sin(yaw)*v)/sqrt(px*px+py*py);

        Zsig(0,i) = p;
        Zsig(1,i) = delta;
        Zsig(2,i) = p_dot;
    }

    // calculate mean predicted measurement
    z_pred.fill(0.0);
    for(int i = 0; i< (2*n_aug_+1); i++){
        z_pred = z_pred + weights_(i)*Zsig.col(i);
    }

    // calculate innovation covariance matrix S

    S.fill(0.0);
    for(int i = 0; i< 2*n_aug_+1; i++){
        VectorXd zDiff = Zsig.col(i) - z_pred;
        while (zDiff(1)> M_PI) zDiff(1)-=2.*M_PI;
        while (zDiff(1)<-M_PI) zDiff(1)+=2.*M_PI;
        S = S + weights_(i)*zDiff*zDiff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

    S = S + R;


    // UPDATE
    // Cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd  z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    //NIS calculation
    nis_radar_ = z_diff.transpose()*S.inverse()*z_diff;
}