use pasta_curves::{group::{ff::{Field, PrimeField}, Group}, pallas};
use std::{ops::{Add, Mul}, u128};
use crate::generator::{to_binary_vec, to_inverse_vec, generator_arbitrary_vec_with_fix_len, GenerateParameter};
use crate::utils::{extand_exp_vec, inner_product, constract_lr_poly, inner_product_add};

pub struct CommitParameter {
    pub secret_v: u128,
    pub secret_r: u128,
    pub x: pallas::Scalar,
    pub y: pallas::Scalar,
    pub z: pallas::Scalar,
    pub len: usize,
    pub generate_parameter: GenerateParameter
}

pub struct Commit{
    pub commit_a: pasta_curves::Ep, 
    pub commit_s: pasta_curves::Ep, 
    pub commit_t1: pasta_curves::Ep, 
    pub commit_t2: pasta_curves::Ep
}

pub struct Response {
    pub response_l: Vec<pasta_curves::Fq>,
    pub response_r: Vec<pasta_curves::Fq>,
    pub response_t: pasta_curves::Fq,
    pub response_tau: pasta_curves::Fq,
    pub response_u: pasta_curves::Fq
}

impl CommitParameter {
    pub fn proof(&self) -> (Commit, Response) {
        let g_vec = self.generate_parameter.g_vec.clone();
        let h_vec = self.generate_parameter.h_vec.clone();
        let h_single_point = self.generate_parameter.h_single_point;

        let al = to_binary_vec(self.secret_v, self.len);
        let ar = to_inverse_vec(al.clone(), self.len);
        let alpha = pallas::Scalar::random(rand::rngs::OsRng);
        let commit_a = constract_commit(
        h_single_point,
            alpha, 
            g_vec.clone(),
            al.clone(),
            h_vec.clone(), 
            ar.clone(), 
            self.len);
        
        let p = pallas::Scalar::random(rand::rngs::OsRng);
        let sl = generator_arbitrary_vec_with_fix_len(self.len);
        let sr = generator_arbitrary_vec_with_fix_len(self.len);
        let commit_s = constract_commit(h_single_point, p, g_vec.clone(),sl.clone(),h_vec.clone(), sr.clone(), self.len);
        
        //hash
        let x = self.x;
        let y = self.y;
        let z = self.z;
        

        let (l_poly, r_poly) = constract_lr_poly(y.clone(), z.clone(), al.clone(), ar.clone(), sl, sr);

        let generator_point = pallas::Point::generator();
        let vg = generator_point.mul(pallas::Scalar::from_u128(self.secret_v));
        let rh = h_single_point.mul(pallas::Scalar::from_u128(self.secret_r));
        let commit = vg + rh;
        let poly_t = inner_product(l_poly.clone(), r_poly.clone());

        let exp_1_vec = extand_exp_vec(pallas::Scalar::one(), self.len);
        let exp_y_vec = extand_exp_vec(y, self.len);
        let exp_2_vec = extand_exp_vec(pallas::Scalar::from_u128(2), self.len);
        let delta_yz = (z - z.square()) * inner_product_add(exp_1_vec.clone(), exp_y_vec.clone()) - z.square() * z * inner_product_add(exp_1_vec, exp_2_vec.clone());
        let t_poly_check = z * z * pallas::Scalar::from_u128(self.secret_v) + delta_yz;
        assert_eq!(t_poly_check, poly_t[0]);
        
        let tau1 = pallas::Scalar::random(rand::rngs::OsRng);
        let tau2 = pallas::Scalar::random(rand::rngs::OsRng);

        let t1g = generator_point.mul(poly_t[1]);
        let tau1h = h_single_point.mul(tau1);
        let commit_t1 = t1g.add(tau1h);

        let t2g = generator_point.mul(poly_t[2]);
        let tau2h = h_single_point.mul(tau2);
        let commit_t2 = t2g.add(tau2h);

        let response_l = input_parameter_to_poly(l_poly, x);
        let response_r = input_parameter_to_poly(r_poly, x);
        let response_t = inner_product_add(response_l.clone(), response_r.clone());
        let response_tau = tau2 * x.square() + tau1 * x + z.square() * pallas::Scalar::from_u128(self.secret_r);
        let response_u = alpha + p * x;
        
        let commit: Commit = Commit {
            commit_a,
            commit_s,
            commit_t1,
            commit_t2,
        };

        let response: Response = Response{
            response_l,
            response_r,
            response_t,
            response_tau,
            response_u
        };

        (commit, response)
    }
}

//rm pub
pub fn constract_commit(
    point_single: pallas::Point, 
    scalar_single: pallas::Scalar, 
    point_vec1: Vec<pallas::Point>,
    scalar_vec1: Vec<pallas::Scalar>,
    point_vec2: Vec<pallas::Point>,
    scalar_vec2: Vec<pallas::Scalar>,
    len:usize
) -> pallas::Point {
    assert_eq!(point_vec1.len(), point_vec2.len(),"poly's len unequal");
    assert_eq!(scalar_vec1.len(), scalar_vec2.len(),"poly's len unequal");
    assert_eq!(point_vec1.len(), scalar_vec1.len(),"poly's len unequal");

    let mut out = point_single.mul(scalar_single);
    for i in 0..len {
        out = out.add(point_vec1[i].mul(scalar_vec1[i]));
        out = out.add(point_vec2[i].mul(scalar_vec2[i]));
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