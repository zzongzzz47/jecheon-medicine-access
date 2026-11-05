"""Mixed-facility siting using walking-distance reductions, without arbitrary scores.

All baseline distances must be finite. Unconnected or baseline-unreachable demand
must be audited separately by the caller, never silently assigned zero distance.
Population and elderly demand can occupy separate rows with their own weights.
"""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack, csr_matrix

def solve_siting(baseline_m, candidate_m, population, max_sites, *, elderly=None,
                 retain_efficiency=None, installation_cost=None, budget=None, time_limit=120):
    baseline=np.asarray(baseline_m,float);distance=np.asarray(candidate_m,float);weights=np.asarray(population,float)
    if distance.ndim!=2 or distance.shape[0]!=len(baseline) or len(weights)!=len(baseline):raise ValueError('Demand/distance dimensions disagree')
    if not np.isfinite(baseline).all() or (baseline<0).any():raise ValueError('Audit disconnected demand; baseline distances must be finite and nonnegative')
    if not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('Population weights must be nonnegative with positive sum')
    if np.isnan(distance).any() or (distance<0).any():raise ValueError('Distances must be nonnegative; unreachable candidate paths use inf')
    n,m=distance.shape
    if not 0<=max_sites<=m:raise ValueError('Invalid site count')
    elder=np.zeros(n) if elderly is None else np.asarray(elderly,float)
    if elder.shape!=(n,) or not np.isfinite(elder).all() or (elder<0).any():raise ValueError('Invalid elderly weights')
    gain=baseline[:,None]-distance
    oi,cj=np.where((gain>1e-7)&((weights+elder)[:,None]>0));benefit=gain[oi,cj];q=len(oi)
    # x_j binary installations, u_ij continuous demand assignments; unassigned = existing baseline.
    rr=np.r_[oi,n+np.arange(q),n+np.arange(q),np.full(m,n+q)]
    cc=np.r_[m+np.arange(q),m+np.arange(q),cj,np.arange(m)]
    vv=np.r_[np.ones(q),np.ones(q),-np.ones(q),np.ones(m)]
    A=coo_matrix((vv,(rr,cc)),shape=(n+q+1,m+q)).tocsr()
    lb=np.full(n+q+1,-np.inf);ub=np.r_[np.ones(n),np.zeros(q),max_sites]
    if budget is not None:
        cost=np.ones(m) if installation_cost is None else np.asarray(installation_cost,float)
        if cost.shape!=(m,) or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Invalid costs')
        A=vstack([A,csr_matrix(np.r_[cost,np.zeros(q)][None,:])],format='csr');lb=np.r_[lb,-np.inf];ub=np.r_[ub,budget]
    total=np.r_[np.zeros(m),weights[oi]*benefit]
    old=np.r_[np.zeros(m),elder[oi]*benefit]
    integral=np.r_[np.ones(m),np.zeros(q)]
    def run(objective,mat,lower,upper):
        scale=max(1.,float(np.max(objective)))
        result=milp(-objective/scale,integrality=integral,bounds=Bounds(np.zeros(m+q),np.ones(m+q)),
            constraints=LinearConstraint(mat,lower,upper),options={'time_limit':time_limit,'mip_rel_gap':1e-7})
        if result.x is None:raise RuntimeError('No feasible solution: '+result.message)
        chosen=np.flatnonzero(result.x[:m]>.5)
        final=np.minimum(baseline,np.min(distance[:,chosen],axis=1)) if len(chosen) else baseline.copy()
        return dict(selected_indices=chosen.tolist(),population_distance_reduction_person_m=float(weights@(baseline-final)),
            elderly_distance_reduction_person_m=float(elder@(baseline-final)),
            population_weighted_mean_m=float(weights@final/weights.sum()),
            elderly_weighted_mean_m=float(elder@final/elder.sum()) if elder.sum()>0 else None,
            optimal=bool(result.success),solver_status=int(result.status),message=result.message,
            mip_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None)
    efficient=run(total,A,lb,ub)
    if retain_efficiency is None:return {'efficiency':efficient}
    if not 0<=retain_efficiency<=1 or elder.sum()<=0:raise ValueError('Need elderly weights and efficiency retention in [0,1]')
    if not efficient['optimal']:raise RuntimeError('Efficiency optimum not certified; do not claim optimum-relative retention')
    floor=retain_efficiency*efficient['population_distance_reduction_person_m']
    scale=max(1.,float(np.max(total)))
    A2=vstack([A,csr_matrix(total[None,:]/scale)],format='csr')
    fair=run(old,A2,np.r_[lb,floor/scale],np.r_[ub,np.inf])
    assert fair['population_distance_reduction_person_m']+1e-4>=floor
    return {'efficiency':efficient,'elderly_priority':fair,'efficiency_retention':retain_efficiency}
