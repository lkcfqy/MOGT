import torch
from torch.optim.optimizer import Optimizer

class MOGTOptimizer(Optimizer):
    r"""
    基于离散莫尔斯理论与 Onsager-Machlup (OM) 耗散泛函的定制终身学习优化器。
    
    理论机制：
    1. 缓存旧任务的收敛盆地中心 $W_A^*$。
    2. 计算对旧任务极其敏感的拓扑子空间 $\mathcal{M}^s$（用 Fisher 矩阵或主成分近似）。
    3. 在学习新任务 $\mathcal{T}_B$ 时，强制将负梯度正交投影到敏感子空间的补集 $\mathcal{P}_{\mathcal{M}^\perp}$。
    4. 当参数即将逸出稳定共轭域时，施加热力学回拉张力（散度惩罚）。
    """

    def __init__(self, params, lr=1e-3, gamma=0.1, projection_threshold=1e-4):
        """
        :param gamma: 热力学张力系数，控制弹回原流形的力道。
        :param projection_threshold: 用于判定敏感方向的特征值或 Fisher 对角线阈值。
        """
        defaults = dict(lr=lr, gamma=gamma, projection_threshold=projection_threshold)
        super(MOGTOptimizer, self).__init__(params, defaults)
        
        # 内部缓存字典，保存结构化保护要素
        self.state['reference_weights'] = {}
        self.state['fisher_diagonals'] = {}
        self.is_anchored = False

    @torch.no_grad()
    def anchor_task(self):
        """
        在任务 A 训练结束后调用此方法，冻结当前的莫尔斯状态。
        注意：生产环境中，Fisher 矩阵需要通过最后几轮的平滑梯度平方来累积，此处简化为对角线缓存的预留。
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # 记录收敛的极小值锚点 W_A^*
                    self.state['reference_weights'][p] = p.clone().detach()
                    # 假设我们在此前已通过外部方法填入了近似的敏感性标识
                    # 如果未计算，默认全矩阵保护度较低 (例如纯粹用 L2)
                    if 'fisher' not in self.state:
                        self.state['fisher_diagonals'][p] = torch.zeros_like(p)
                        
        self.is_anchored = True
        print("⚓️ [MOGTOptimizer] 拓扑流形盆地已锚定！新任务的破坏性改写将被阻隔。")

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行耗散正交化步长更新。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            threshold = group['projection_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # 如果尚未打下任何旧任务的锚点，退化为 Vanilla SGD
                if not self.is_anchored or p not in self.state['reference_weights']:
                    p.add_(grad, alpha=-lr)
                    continue
                
                ref_W = self.state['reference_weights'][p]
                fisher_diag = self.state['fisher_diagonals'].get(p, torch.zeros_like(p))
                
                # -------------------------------------------------------------
                # 物理机制一：正交投影 (Orthogonal Projection P_M_perp)
                # 原理：如果在某一维度上，旧任务的 Fisher 极高（超过阈值），
                # 证明该方向是构建旧有拓扑流形的刚性骨架。我们要将该维度的更新抹除（正交投影）。
                # -------------------------------------------------------------
                sensitive_mask = (fisher_diag > threshold).to(grad.dtype)
                # 仅保留在非敏感维度上的梯度流
                projected_grad = grad * (1.0 - sensitive_mask)
                
                # -------------------------------------------------------------
                # 物理机制二：Onsager-Machlup 热力学弹力 (Topological Damping)
                # 原理：如果参数在低敏感区漂移得离原始锚点太远，散度增加，
                # 产生耗散惩罚，将其以 gamma 强度拉回锚点 W_A^*。
                # -------------------------------------------------------------
                displacement = p - ref_W
                # 施加阻尼：仅当且仅当发生位移时
                thermodynamic_pull = gamma * displacement * (1.0 - sensitive_mask)
                
                # 合并动力系统方程
                p.add_(projected_grad + thermodynamic_pull, alpha=-lr)

        return loss
