import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.linalg import eig

# 定义平衡点方程
def equilibrium(state, tau_E, tau_I, M_EE, M_EI, M_IE, M_II, h_E, h_I):
    nu_E, nu_I = state
    z_E = M_EE * nu_E - M_EI * nu_I + h_E
    z_I = M_IE * nu_E - M_II * nu_I + h_I
    # 仅考虑 [z]^+ > 0 的情况
    if z_E < 0 or z_I < 0:
        return [1e3, 1e3]  # 返回一个大的误差，避免非激活状态
    eq1 = -nu_E + M_EE * nu_E - M_EI * nu_I + h_E
    eq2 = -nu_I + M_IE * nu_E - M_II * nu_I + h_I
    return [eq1, eq2]

# 定义雅可比矩阵
def jacobian(M_EE, M_EI, M_IE, M_II, tau_E, tau_I):
    J = np.array([
        [(-1 + M_EE)/tau_E, -M_EI/tau_E],
        [M_IE/tau_I, (-1 - M_II)/tau_I]
    ])
    return J

# 定义微分方程
def neural_network(state, t, tau_E, tau_I, M_EE, M_EI, M_IE, M_II, h_E, h_I):
    nu_E, nu_I = state
    # 激活函数内的表达式
    z_E = M_EE * nu_E - M_EI * nu_I + h_E
    z_I = M_IE * nu_E - M_II * nu_I + h_I
    # 应用非线性激活函数 [z]^+
    f_E = max(z_E, 0)
    f_I = max(z_I, 0)
    # 微分方程
    dnu_E_dt = (-nu_E + f_E) / tau_E
    dnu_I_dt = (-nu_I + f_I) / tau_I
    return [dnu_E_dt, dnu_I_dt]

def main():
    # 参数设置
    tau_E = 1.0  # 兴奋性神经元的时间常数
    tau_I = 2.0  # 抑制性神经元的时间常数
    M_EE = 1.5    # 兴奋性到兴奋性突触强度
    M_EI = 1.0    # 抑制性反馈突触强度
    M_IE = 1.0    # 抑制性到兴奋性突触强度
    M_II = 0.0    # 抑制性到抑制性突触强度
    h_E = 1.0     # 兴奋性神经元接收的外部刺激
    h_I = 2.0     # 抑制性神经元接收的外部刺激
    
    # 初始猜测
    initial_guess = [1.0, 1.0]
    
    # 求解平衡点
    nu_E_star, nu_I_star = fsolve(equilibrium, initial_guess, args=(tau_E, tau_I, M_EE, M_EI, M_IE, M_II, h_E, h_I))
    print(f"equilibrium: nu_E* = {nu_E_star:.3f}, nu_I* = {nu_I_star:.3f}")
    
    # 计算雅可比矩阵
    J = jacobian(M_EE, M_EI, M_IE, M_II, tau_E, tau_I)
    print(f"Jacobi matrix J:\n{J}")
    
    # 计算特征值
    eigenvals = eig(J)[0]
    print(f"eigen values: {eigenvals}")
    
    # 计算Trace和Determinant
    Trace_J = np.trace(J)
    Det_J = np.linalg.det(J)
    print(f"Trace(J) = {Trace_J}")
    print(f"Det(J) = {Det_J}")
    discriminant = Trace_J**2 - 4*Det_J
    print(f"discriminant (Trace(J)^2 - 4*Det(J)) = {discriminant}")
    
    # 判断是否满足Hopf分岔条件
    if np.isclose(Trace_J, 0, atol=1e-3) and discriminant <0:
        print("system satisfies the hopf condition")
    else:
        print("system doesnt satisfy the hopf condition")
    
    # 模拟系统动态
    initial_state = [2.0, 2.0]  # 平衡点附近的初始条件
    t = np.linspace(0, 100, 2000)
    solution = odeint(neural_network, initial_state, t, args=(tau_E, tau_I, M_EE, M_EI, M_IE, M_II, h_E, h_I))
    nu_E = solution[:,0]
    nu_I = solution[:,1]
    
    # 绘制时间序列
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    plt.plot(t, nu_E, label=r'$\nu_E$')
    plt.plot(t, nu_I, label=r'$\nu_I$')
    plt.axhline(nu_E_star, color='blue', linestyle='--', alpha=0.5, label=r'$\nu_E^*$')
    plt.axhline(nu_I_star, color='orange', linestyle='--', alpha=0.5, label=r'$\nu_I^*$')
    plt.xlabel('time')
    plt.ylabel('rate')
    plt.title('Time series of Hopf bifurcation points')
    plt.legend()
    
    # 绘制相图
    plt.subplot(1,2,2)
    plt.plot(nu_E, nu_I)
    plt.plot(nu_E_star, nu_I_star, 'ro', label='equilibrium')
    plt.xlabel(r'$\nu_E$')
    plt.ylabel(r'$\nu_I$')
    plt.title('Phase diagram of Hopf bifurcation points')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
