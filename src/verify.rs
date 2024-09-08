use pasta_curves::{group::{ff::{Field, PrimeField}, Group}, pallas};
use std::ops::{Add, Mul};
use crate::{generator, utils::{extand_exp_vec, inner_product_add}};
use crate::proof::{Commit, Response};

pub struct VerifyParameter{
    pub commit_origin: pallas::Point,
    pub commit: Commit,
    pub response: Response,
    pub x: pallas::Scalar, 
    pub g_vec: Vec<pallas::Point>,
    pub z: pallas::Scalar, 
    pub h_vec: Vec<pallas::Point>,
    pub y: pallas::Scalar, 
    pub len: usize,
    pub h_single_point: pallas::Point
}

impl VerifyParameter {
    pub fn verify(&self) {
        let commit_p = compute_commit_p(
            &self.commit.commit_a,
            &self.commit.commit_s,
            &self.x,
            &self.g_vec,
            &self.z,
            &self.h_vec,
            &self.y
        );

        let exp_1_vec = extand_exp_vec(pallas::Scalar::one(), self.len);
        let exp_y_vec = extand_exp_vec(self.y, self.len);
        let exp_2_vec = extand_exp_vec(pallas::Scalar::one().double(), self.len);
        let delta_yz = (self.z - self.z.square()) * inner_product_add(exp_1_vec.clone(), exp_y_vec.clone()) - self.z.cube() * inner_product_add(exp_1_vec, exp_2_vec);
        
        let generator_point = pallas::Point::generator();
        let verify_equation1_left = generator_point.mul(self.response.response_t).add(self.h_single_point.mul(self.response.response_tau));
        let verify_equation1_right = self.commit_origin.mul(self.z * self.z).add(generator_point.mul(delta_yz)).add(self.commit.commit_t1.mul(self.x)).add(self.commit.commit_t2.mul(self.x.square()));
        let verify_check1 = pallas::Point::eq(&verify_equation1_left, &verify_equation1_right);
        println!("check1: {}", verify_check1);
        assert_eq!(verify_check1, true);

        let mut verify_equation2 = self.h_single_point.mul(self.response.response_u);
        let mut inv_yh_vec = self.h_vec.clone();
        let inv_y = self.y.invert().unwrap();
        let mut scalar = inv_y;
        for i in 1..inv_yh_vec.len(){
            inv_yh_vec[i] = inv_yh_vec[i].mul(scalar);
            scalar = scalar * inv_y;
        }
        for i in 0..self.len{
            verify_equation2 = verify_equation2.add(self.g_vec[i].mul(self.response.response_l[i]));
            verify_equation2 = verify_equation2.add(inv_yh_vec[i].mul(self.response.response_r[i]));
        }
        let check2 = pallas::Point::eq(&verify_equation2, &commit_p);
        println!("check2: {}", check2);
        assert_eq!(check2, true);
        
        let mut out = pallas::Point::identity();
        let mut out2 = pallas::Point::identity();
        for i in 0..self.g_vec.len(){
            out = out.add(inv_yh_vec[i].mul(self.z * exp_y_vec[i]));
            out2 = out2.add(self.h_vec[i].mul(self.z));
        }
        let check3 = pallas::Point::eq(&out, &out2);
        println!("check3: {}", check3);
        assert_eq!(check3, true);
    }

    pub fn verify_range_proof(&self, low_commit: Commit, high_commit: Commit, low_response: Response, high_response: Response, range_low: u128, range_high: u128) {
        //check commit is correct
        let generator = pallas::Point::generator();
        let p1 = generator.mul(pallas::Scalar::from_u128(range_low).neg());//-low
        let p2 = generator.mul(pallas::Scalar::from_u128(range_high).neg());//-high
        //p3 = v-low (should be inside 2^len)
        let p3 = self.commit_origin.add(p1);
        let exp2 = (1u128 << self.len) as u128; 
        let p4 = generator.mul(pallas::Scalar::from_u128(exp2));
        //p5 = 2^len + v - high
        let p5 = self.commit_origin.add(p4).add(p2);

        let commit_check1 = pallas::Point::eq(&low_commit.commit_origin, &p3);
        let commit_check2 = pallas::Point::eq(&high_commit.commit_origin, &p5);
        assert!(commit_check1, "low commit isn't correct");
        assert!(commit_check2, "high commit isn't correct");


        //verify proof
        let low_Parameter = VerifyParameter{
            commit_origin: low_commit.commit_origin,
            commit: low_commit,
            response: low_response,
            x: self.x, 
            g_vec: self.g_vec.clone(),
            z: self.z, 
            h_vec: self.h_vec.clone(),
            y: self.y, 
            len: self.len,
            h_single_point: self.h_single_point
        };

        let high_Parameter = VerifyParameter{
            commit_origin: high_commit.commit_origin,
            commit: high_commit,
            response: high_response,
            x: self.x, 
            g_vec: self.g_vec.clone(),
            z: self.z, 
            h_vec: self.h_vec.clone(),
            y: self.y, 
            len: self.len,
            h_single_point: self.h_single_point
        };

        self.verify();
        low_Parameter.verify();
        high_Parameter.verify();
    }
}


fn compute_commit_p (
    commit_a: &pallas::Point, 
    commit_s: &pallas::Point, 
    x: &pallas::Scalar, 
    g_vec: &Vec<pallas::Point>,
    z: &pallas::Scalar, 
    h_vec: &Vec<pallas::Point>,
    y: &pallas::Scalar, 
    ) -> pallas::Point {
    let mut out = *commit_a;
    let mut inv_yh_vec = h_vec.clone();
    let inv_y = y.invert().unwrap();
    let mut inv_y_scalar = inv_y;
    for i in 1..h_vec.len(){
        inv_yh_vec[i] = inv_yh_vec[i].mul(inv_y_scalar);
        inv_y_scalar = inv_y_scalar * inv_y;
    }

    let neg_z = z.neg();
    let exp_y_vec = extand_exp_vec(*y, h_vec.len());
    let exp_2_vec = extand_exp_vec(pallas::Scalar::one().double(), h_vec.len());
    out = out.add(commit_s.mul(x));
    for i in 0..g_vec.len(){
        out = out.add(g_vec[i].mul(neg_z));
        out = out.add(inv_yh_vec[i].mul(z * exp_y_vec[i] + z.square() * exp_2_vec[i]));
    }
    out
}

fn input_parameter_to_poly (poly: Vec<[pallas::Scalar;2]>, x: pallas::Scalar) -> Vec<pallas::Scalar> {
    let mut out = Vec::new();
    for i in 0..poly.len(){
        out.push(poly[i][0] + poly[i][1] * x);
    }
    out
}