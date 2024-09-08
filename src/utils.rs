use pasta_curves::{group::ff::PrimeField, pallas};

pub fn extand_exp_vec (num: pallas::Scalar, len: usize) -> Vec<pallas::Scalar> {
    let mut out = vec![pallas::Scalar::one();len];
    for i in 1..len{
        out[i] = out[i-1] * num; 
    }
    out
}

pub fn inner_product(
    l_poly: Vec<[pallas::Scalar;2]>,
    r_poly: Vec<[pallas::Scalar;2]>
) -> Vec<pallas::Scalar> {
     let mut out = vec![pallas::Scalar::zero(); 3];
     for i in 0..l_poly.len(){
         out[0] = out[0] + l_poly[i][0] * r_poly[i][0];
         out[1] = out[1] + l_poly[i][1] * r_poly[i][0];
         out[1] = out[1] + l_poly[i][0] * r_poly[i][1];
         out[2] = out[2] + l_poly[i][1] * r_poly[i][1];
     }
     out
}

pub fn constract_lr_poly(
    y: &pallas::Scalar,
    z: &pallas::Scalar,
    al: &Vec<pallas::Scalar>,
    ar: &Vec<pallas::Scalar>,
    sl: &Vec<pallas::Scalar>,
    sr: &Vec<pallas::Scalar>,
) -> (Vec<[pallas::Scalar; 2]>, Vec<[pallas::Scalar; 2]>) {
    let mut l_poly: Vec<[pallas::Scalar; 2]> = Vec::with_capacity(al.len());
    let mut r_poly: Vec<[pallas::Scalar; 2]> = Vec::with_capacity(ar.len());

    let two = pallas::Scalar::from_u128(2);
    let mut exp_2 = pallas::Scalar::one();
    let mut exp_y = pallas::Scalar::one();

    for i in 0..al.len() {
        l_poly.push([al[i] - *z, sl[i]]);
        r_poly.push([
            exp_y * (ar[i] + *z) + z * z * exp_2,
            exp_y * sr[i],
        ]);

        exp_2 = exp_2 * two;
        exp_y = exp_y * y;
    }

    (l_poly, r_poly)
}

pub fn inner_product_add (poly1: Vec<pallas::Scalar>, poly2: Vec<pallas::Scalar>) -> pallas::Scalar {
    assert_eq!(poly1.len(), poly2.len(),"poly's len unequal");
    let mut out = pallas::Scalar::zero();
    for i in 0..poly1.len(){
        out = out + (poly1[i] * poly2[i]);
    }
    out
}