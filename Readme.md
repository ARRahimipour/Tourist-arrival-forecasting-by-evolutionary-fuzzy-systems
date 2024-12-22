# Fuzzy Genetic Prediction System

این پروژه از منطق فازی برای پیش‌بینی تعداد ورودها در ماه جاری استفاده می‌کند، بر اساس تعداد ورودهای ماه گذشته و دو ماه پیش. همچنین، الگوریتم ژنتیک برای بهینه‌سازی قوانین فازی به‌کار رفته است تا دقت پیش‌بینی بهبود یابد.

## مراحل اجرای کد

### مرحله 1: تعریف متغیرهای فازی و توابع عضویت

در این مرحله، متغیرهای فازی برای تعداد ورودهای ماه گذشته (`arrival_last_month`)، تعداد ورودهای دو ماه پیش (`arrival_two_months_ago`)، و پیش‌بینی تعداد ورودهای ماه جاری (`predicted_arrival`) تعریف می‌شوند. برای هر کدام از این متغیرها، توابع عضویت مثلثی تعریف می‌شود که مقادیر ورودی را به سه دسته کم، متوسط و زیاد تقسیم می‌کند.

```python
arrival_last_month['low'] = fuzz.trimf(arrival_last_month.universe, [0, 0, 50])
arrival_last_month['medium'] = fuzz.trimf(arrival_last_month.universe, [0, 50, 100])
arrival_last_month['high'] = fuzz.trimf(arrival_last_month.universe, [50, 100, 100])

arrival_two_months_ago['low'] = fuzz.trimf(arrival_two_months_ago.universe, [0, 0, 50])
arrival_two_months_ago['medium'] = fuzz.trimf(arrival_two_months_ago.universe, [0, 50, 100])
arrival_two_months_ago['high'] = fuzz.trimf(arrival_two_months_ago.universe, [50, 100, 100])

predicted_arrival['low'] = fuzz.trimf(predicted_arrival.universe, [0, 0, 50])
predicted_arrival['medium'] = fuzz.trimf(predicted_arrival.universe, [0, 50, 100])
predicted_arrival['high'] = fuzz.trimf(predicted_arrival.universe, [50, 100, 100])
```

### مرحله 2: تعریف قوانین فازی

در این مرحله، پنج قانون فازی برای تعیین پیش‌بینی تعداد ورودها بر اساس تعداد ورودهای ماه گذشته و دو ماه پیش تعریف می‌شود. این قوانین به کمک عملگرهای منطقی مانند `AND` ترکیب می‌شوند.

```python
rule1 = ctrl.Rule(arrival_last_month['low'] & arrival_two_months_ago['low'], predicted_arrival['low'])
rule2 = ctrl.Rule(arrival_last_month['medium'] & arrival_two_months_ago['medium'], predicted_arrival['medium'])
rule3 = ctrl.Rule(arrival_last_month['high'] & arrival_two_months_ago['high'], predicted_arrival['high'])
rule4 = ctrl.Rule(arrival_last_month['low'] & arrival_two_months_ago['high'], predicted_arrival['medium'])
rule5 = ctrl.Rule(arrival_last_month['high'] & arrival_two_months_ago['low'], predicted_arrival['medium'])
```

### مرحله 3: ایجاد سیستم کنترل فازی

پس از تعریف قوانین، سیستم کنترل فازی ایجاد می‌شود و یک شبیه‌سازی برای پیش‌بینی تعداد ورودها انجام می‌شود.

```python
fuzzy_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_control_system)
```

### مرحله 4: پیش‌بینی تعداد ورودها

تابع `forecast` برای پیش‌بینی تعداد ورودها استفاده می‌شود. این تابع ورودی‌هایی از تعداد ورودهای ماه گذشته و دو ماه پیش می‌گیرد و پیش‌بینی تعداد ورودها برای ماه جاری را محاسبه می‌کند.

```python
def forecast(arrival_last, arrival_two_months):
    fuzzy_simulation.input['arrival_last_month'] = arrival_last
    fuzzy_simulation.input['arrival_two_months_ago'] = arrival_two_months
    fuzzy_simulation.compute()
    return fuzzy_simulation.output['predicted_arrival']
```

### مرحله 5: بهینه‌سازی قوانین با الگوریتم ژنتیک

در این مرحله، از الگوریتم ژنتیک برای بهینه‌سازی قوانین فازی استفاده می‌شود. هدف این است که پارامترهای مدل به گونه‌ای تنظیم شوند که خطای پیش‌بینی (MSE) کمینه شود.

```python
def fitness(individual):
    # داده‌های نمونه برای ارزیابی مدل
    sample_data = [(20, 30, 25), (70, 80, 75), (50, 50, 50)]
    mse = 0  # مقدار میانگین مربع خطا
    for arrival_last, arrival_two_months, actual in sample_data:
        prediction = forecast(arrival_last, arrival_two_months)
        mse += (prediction - actual) ** 2
    return (mse / len(sample_data)),
```

### مرحله 6: تنظیمات الگوریتم ژنتیک

برای بهینه‌سازی با الگوریتم ژنتیک، جمعیتی از افراد به صورت تصادفی ایجاد می‌شود و از عملگرهای تقاطع (`mate`) و جهش (`mutate`) استفاده می‌شود تا بهترین فرد (مدل بهینه‌شده) پیدا شود.

```python
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=10)  # ایجاد جمعیت اولیه
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)  # اجرای الگوریتم ژنتیک
```

### مرحله 7: نمایش بهترین نتیجه

پس از اجرای الگوریتم ژنتیک، بهترین فرد که کمترین خطا را دارد، نمایش داده می‌شود.

```python
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
```

## توضیحات
1. **تعریف متغیرهای فازی:**  
   برای پیش‌بینی تعداد ورودها از متغیرهای فازی استفاده می‌شود که برای هر یک از آن‌ها توابع عضویت مثلثی تعریف شده است.
2. **تعریف قوانین فازی:**  
   پنج قانون فازی برای پیش‌بینی تعداد ورودها بر اساس تعداد ورودهای ماه گذشته و دو ماه پیش تنظیم شده‌اند.

3. **سیستم کنترل فازی:**  
   سیستم فازی به کمک قوانین تعریف شده ایجاد می‌شود و برای انجام پیش‌بینی‌ها از شبیه‌سازی استفاده می‌شود.

4. **الگوریتم ژنتیک برای بهینه‌سازی:**  
   از الگوریتم ژنتیک برای بهینه‌سازی قوانین فازی استفاده می‌شود تا با کاهش خطا، مدل پیش‌بینی بهبود یابد.

## نصب پیش‌نیازها

برای اجرای این کد، نیاز به نصب بسته‌های زیر دارید:

```bash
pip install numpy scikit-fuzzy deap
```

## نتیجه نهایی
خروجی این کد شامل موارد زیر خواهد بود:

1. پیش‌بینی تعداد ورودها برای ماه جاری بر اساس تعداد ورودهای ماه گذشته و دو ماه پیش.
2. بهترین مدل بهینه‌شده با استفاده از الگوریتم ژنتیک برای کاهش خطای پیش‌بینی.

این پروژه می‌تواند به عنوان یک ابزار پیش‌بینی در مسائل تجاری مختلف مانند پیش‌بینی تعداد ورود مشتریان به فروشگاه یا پیش‌بینی تعداد درخواست‌ها در یک سیستم آنلاین استفاده شود.
