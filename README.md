# WARP_implementation

### Abstract
Метод Reinforcement Learning from Human Feedback (RLHF) популярный подход для алаймента языковых моделей, однако его реализация, особенно с использованием Proximal Policy Optimization (PPO), вызывает множество сложностей из-за нестабильности процесса обучения. В статье Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs был предложен более простой подход на основе метода REINFORCE, который в ряде задач демонстрирует лучшие результаты. В работе WARP: On the Benefits of Weight Averaged Rewarded Policies представлена модификация метода REINFORCE с использованием техник объединения моделей, что может привести к более стабильному и эффективному обучению. В данной работе приведена реализация и анализ этого метода.

### Quick start for tbank reviewers
- [быстрый запуск метода](https://colab.research.google.com/drive/1nO5wbPVXitXNT6sg-X8o-8VYhqB32rSI?usp=sharing), рекомендуется использовать в google colab

### Variating EMA coefficient $\mu$
- [оценка влияния гиперпараметра на результаты модели](https://colab.research.google.com/drive/1QGTrRP_8WBRIOTq_i5ECwiX0HJ_hGWWI?usp=sharing)

### Getting started with local machine

- Клонирование репозитория
```bash
git clone https://github.com/Arslan203/WARP_implementation.git
cd WARP_implementation
```

- Установка зависимостей. (Python 3 + NVIDIA GPU + CUDA)
```bash
pip install -r requirements.txt
```

- Обучение модели наград
```bash
python reward_trainer/main.py --num_epochs 4 --batch_per_epoch 1000 --train_batch_size 16 --eval_batch_size 64 --save_path reward_model_dir
```
Все параметры для скрипта можно посмотреть в [reward_trainer/config.json](https://github.com/Arslan203/WARP_implementation/blob/main/reward_trainer/config.json)

- WARP метод
```bash
!python WARP/main.py --reward_model ChokeGM/reward_model_imdb --I 2 --M 2 --T 100 --batch_size 32 --mu 0.01 --save_path warp_model_dir
```
Также все параметры лежат в [WARP/config.json](https://github.com/Arslan203/WARP_implementation/blob/main/WARP/config.json)
