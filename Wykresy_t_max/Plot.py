import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# -----------------------------
# GLOBALNE ZMIENNE
# -----------------------------
lambda_1 = 0.00002
lambda_2 = 0.0000006
lambda_3 = 0.00006
lambda_4 = 0.0000012
lambda_5 = 0.0001
lambda_6 = 0.0000018
lambda_7 = 0.00014
lambda_8 = 0.0000024
lambda_9 = 0.00018

# Wybór wykresów do rysowania: wpisz numery pochodnych z listy poniżej (1-7)
wykresy_do_rysowania = [1, 4, 6] # nieparzyste elementy systemu
# wykresy_do_rysowania = [2, 3, 5, 7] # parzyste elementy systemu

# -----------------------------
# FUNKCJE POCZODNYCH
# -----------------------------
def eta_prime_1(t):
    return -(lambda_9 - lambda_1) * np.exp(-(lambda_9 - lambda_1) * t) + lambda_9 * np.exp(-lambda_9 * t)

def eta_prime_2(t):
    return (-lambda_9 + 2*lambda_2*t) * np.exp(-lambda_9*t + lambda_2*t**2) + lambda_9 * np.exp(-lambda_9*t)

def eta_prime_3_4(t):
    return (-lambda_9 + 2*lambda_2*t) * np.exp(-lambda_9*t + lambda_2*t**2) + lambda_9 * np.exp(-lambda_9*t)

def eta_prime_5(t):
    return -(lambda_9 - lambda_5) * np.exp(-(lambda_9 - lambda_5) * t) + lambda_9 * np.exp(-lambda_9*t)

def eta_prime_6(t):
    return (2*lambda_6*t - lambda_9) * np.exp(lambda_6*t**2 - lambda_9*t) + lambda_9 * np.exp(-lambda_9*t)

def eta_prime_7(t):
    return -(lambda_9 - lambda_7) * np.exp(-(lambda_9 - lambda_7)*t) + lambda_9 * np.exp(-lambda_9*t)

def eta_prime_8(t):
    return (2*lambda_8*t - lambda_9) * np.exp(lambda_8*t**2 - lambda_9*t) + lambda_9 * np.exp(-lambda_9*t)

# Lista funkcji do iteracji
pochodne = [eta_prime_1, eta_prime_2, eta_prime_3_4, eta_prime_5, eta_prime_6, eta_prime_7, eta_prime_8]

# Nazwy wykresów
nazwy = [
    "Redundancja elementu nr 1",
    "Redundancja elementu nr 2",
    "Redundancja elementu nr 3 i 4",
    "Redundancja elementu nr 5",
    "Redundancja elementu nr 6",
    "Redundancja elementu nr 7",
    "Redundancja elementu nr 8"
]

# -----------------------------
# FUNKCJA DO ZNAJDOWANIA MIEJSC ZEROWYCH
# -----------------------------
def znajdz_miejsce_zerowe(func, zakres_poszukiwan=(0.1, 15000)):
    """
    Znajduje miejsce zerowe funkcji w podanym zakresie.
    Zwraca None jeśli nie znaleziono miejsca zerowego.
    """
    try:
        # Szukamy miejsca zerowego w podanym zakresie
        t_zero = fsolve(func, zakres_poszukiwan[0])[0]
        
        # Sprawdzamy czy miejsce zerowe jest w rozsądnym zakresie i czy funkcja rzeczywiście zmienia znak
        if zakres_poszukiwan[0] <= t_zero <= zakres_poszukiwan[1]:
            wartosc_w_zerze = func(t_zero)
            if abs(wartosc_w_zerze) < 1e-6:  # tolerancja błędu
                return t_zero
    except:
        pass
    return None

# -----------------------------
# GENEROWANIE WYKRESÓW
# -----------------------------
for idx, func in enumerate(pochodne, start=1):
    if idx not in wykresy_do_rysowania:
        continue

    # Dopasowany zakres t do obserwacji miejsc zerowych
    t_max = 30000
    t = np.linspace(0, t_max, 30000)  # duża gęstość punktów
    y = func(t)

    # Znajdowanie miejsc zerowych
    miejsce_zerowe = znajdz_miejsce_zerowe(func)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label=nazwy[idx-1], linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    
    # Dodanie pionowej linii jeśli znaleziono miejsce zerowe
    if miejsce_zerowe is not None and miejsce_zerowe <= t_max:
        plt.axvline(x=miejsce_zerowe, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, 
                   label=f't* = {miejsce_zerowe:.1f} h')
        # Zaznaczenie punktu na wykresie
        plt.plot(miejsce_zerowe, 0, 'ro', markersize=6)
    else:
        plt.plot([], [], ' ', label='t* → +∞')
    
    plt.xlabel("t [h]")
    plt.ylabel("η'(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"{nazwy[idx-1]} - Wykres pochodnej")
    
    # Dopasowanie zakresu y dla lepszej widoczności
    y_max = np.max(np.abs(y)) * 1.1
    plt.ylim(-y_max, y_max)
    
    plt.tight_layout()
    plt.show()
    
    # Informacja o miejscu zerowym w konsoli
    if miejsce_zerowe is not None:
        print(f"{nazwy[idx-1]}: Miejsce zerowe znalezione przy t = {miejsce_zerowe:.2f}")
    else:
        print(f"{nazwy[idx-1]}: Nie znaleziono miejsca zerowego w rozsądnym zakresie")
    print("---")