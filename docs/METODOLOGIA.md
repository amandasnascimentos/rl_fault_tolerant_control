
# Metodologia (notas factuais)

## Simulador e veículo

- HoloOcean (cliente local `holoocean_rep/client`, versão e commit em `metadata.json`).
- Cenário SimpleUnderwater, veículo BlueROV2, `control_scheme` 0 (Thruster Forces), 60 ticks/s.
- Comando u em newtons. Escala K = 1 N por unidade, obtida por teste de aceleração (ver seção de calibração).

## Geometria e alocação

- B = K · T[:, ho_to_model] · sinais, com T montada a partir da Tabela 4.2 e da eq. 4.62 de Wu (2018).
- Altura dos horizontais zerada (lz = 0), ajustada após comparação com o simulador. Justifique no texto.
- Mapeamento índice HoloOcean → propulsor da tese:

| HO | Posição                     | Tese |
| -- | ----------------------------- | ---- |
| 0  | vertical dianteiro-direito    | T6   |
| 1  | vertical dianteiro-esquerdo   | T5   |
| 2  | vertical traseiro-esquerdo    | T7   |
| 3  | vertical traseiro-direito     | T8   |
| 4  | horizontal dianteiro-direito  | T2   |
| 5  | horizontal dianteiro-esquerdo | T1   |
| 6  | horizontal traseiro-esquerdo  | T3   |
| 7  | horizontal traseiro-direito   | T4   |

- Validação: sinal e acoplamento dos 8 propulsores e dos 6 graus de liberdade (`valida_matriz.py`, `andar.py`).

## Modelo de falha

- Eficiência η_i ∈ [0, 1] por propulsor (1 = saudável, 0 = falho).
- Força efetiva: u_real = η ⊙ u.
- Matriz efetiva: B_eff = B · diag(η).
- Falha instantânea e determinística. Não modela transiente, degradação gradual nem dinâmica do propulsor.
- A alocação recebe η exato (diagnóstico perfeito). Isso é limite superior para W-PINV e QP.

## Controle de alto nível

- PD em posição (referencial do mundo) e atitude (referencial do corpo).
- Força convertida ao corpo por R^T. Torques em N·m.
- Saturação de tau por eixo (tau_max).
- Referência relativa à pose inicial.
- Ganhos definidos por tentativa e erro. Massa e inércia não foram medidas.

## Alocadores

- pinv: u = B⁺ τ, com escala proporcional se max|u| > u_max.
- wpinv: u = W⁻¹ B_effᵀ (B_eff W⁻¹ B_effᵀ)⁻¹ τ, com W⁻¹ = diag(max(η, ε)). Valor de ε: <preencher com o valor do arquivo></preencher>.
- qp: min ||B_eff u − τ||² + ε||u||², sujeito a u_min ≤ u ≤ u_max, resolvido com lsq_linear (ε = 1e-6).

## Métricas

- Erro de posição: e_p(t) = ||p_ref − p(t)||.
- Erro final: e_p no último passo.
- Tempo de convergência: primeiro t com e_p < 0,1 m.
- Saturação: passo saturado se max|u| ≥ 0,95 u_max. Tempo em saturação = passos × dt.
- Erro de alocação: e_a(t) = ||B_eff u − τ||. Reportado como média, pico e média sem saturação.
- Chattering: c(t) = ||u_k − u_(k−1)||.

## Cenários

- Referência: 8 m em x e 8 m em y, yaw de 45°, duração 30 s, mesmo estado inicial.
- sem_falha, falha_horizontal (HO4), falha_vertical (HO0), falha_dupla_horizontal (HO4 e HO5).
- Observação: HO4 e HO5 são ambos dianteiros.

## Limitações

- Uma execução determinística por cenário e alocador.
- Sem análise estatística entre sementes.
- u_max = 40 N vem da tese (T200 a 16 V) e ainda não foi confirmado no simulador.
- Conclusões válidas para o modelo de falha instantânea e para este veículo/cenário.
