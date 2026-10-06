# Tema 2 ASC - CUDA Merkle Root si Proof of Work

**Student:** Mircea Bianca-Anastasia  
**Grupa:** 333CC  
**Fisiere incluse in arhiva:** `utils.cu`, `README.md`

---

## 1. Prezentare

Aceasta tema implementeaza pe GPU doua etape costisitoare din procesul simplificat de minare al unui bloc:

- obtinerea Merkle root-ului pentru tranzactiile din bloc;
- determinarea unui nonce valid pentru Proof of Work.

Implementarea este realizata in `utils.cu` si foloseste CUDA pentru a distribui munca pe mai multe thread-uri. Partea de CPU ramane responsabila pentru coordonarea apelurilor, copierea datelor si scrierea rezultatului final, iar GPU-ul executa operatiile repetitive de hash-uire.

Solutia foloseste trei idei principale:

1. tranzactiile sunt hash-uite in paralel;
2. arborele Merkle este construit pe niveluri, folosind doua buffere alternate;
3. nonce-urile sunt testate in batch-uri, iar rezultatul valid este selectat folosind operatii atomice.

---

## 2. Elemente importante din cod

In implementare apar urmatoarele constante si structuri globale:

```c
#define MERKLE_THREADS 256
#define NONCE_THREADS 256
#define NONCE_BATCH_SIZE (1ULL << 16)
```

Pentru Merkle root se folosesc 256 de thread-uri per block. Aceeasi dimensiune este folosita si pentru kernelul de nonce. Spatiul de cautare pentru nonce este impartit in batch-uri de `2^16` valori, pentru a limita dimensiunea fiecarei lansari de kernel si pentru a permite verificarea periodica pe host.

Memoria de pe GPU este pastrata in buffere globale:

```c
static BYTE *g_d_transactions = NULL;
static BYTE *g_d_hashes_a = NULL;
static BYTE *g_d_hashes_b = NULL;

static uint32_t *g_d_best_nonce = NULL;
static int *g_d_found = NULL;
```

Aceste buffere sunt reutilizate intre apeluri, pentru a evita costul unor alocari repetate.

---

## 3. Gestionarea memoriei

Pentru alocarea bufferelor folosesc functia auxiliara:

```c
static void ensure_byte_buffer(BYTE **ptr, size_t *capacity, size_t needed)
```

Aceasta verifica daca bufferul existent are deja suficienta capacitate. Daca da, bufferul este reutilizat. Daca nu, vechiul buffer este eliberat si se face o noua alocare cu `cudaMalloc`.

Aceasta abordare este utila deoarece functiile `construct_merkle_root` si `find_nonce` pot fi apelate pentru mai multe blocuri. Fara reutilizarea memoriei, timpul ar fi afectat de apeluri frecvente la `cudaMalloc` si `cudaFree`.

Toate apelurile CUDA sunt verificate cu macro-ul `CUDA_CHECK`, astfel incat eventualele erori sa fie detectate imediat.

---

## 4. Calculul Merkle root-ului

### 4.1. Hash-ul initial al tranzactiilor

Functia `construct_merkle_root` incepe prin copierea tranzactiilor pe GPU:

```c
cudaMemcpy(g_d_transactions, transactions, transactions_bytes, cudaMemcpyHostToDevice)
```

Apoi se lanseaza kernelul:

```c
hash_transactions_kernel<<<blocks, threads>>>(...)
```

Fiecare thread calculeaza hash-ul unei tranzactii. Indexul tranzactiei este obtinut din `blockIdx`, `blockDim` si `threadIdx`.

Se foloseste lungimea `transaction_size - 1`, deoarece dimensiunea tranzactiei include si caracterul nul de final, iar acesta nu trebuie inclus in hash-ul efectiv al tranzactiei.

### 4.2. Reducerea nivelurilor

Dupa hash-uirea tranzactiilor, vectorul de hash-uri este redus pana cand ramane un singur element. Aceasta reducere este facuta cu kernelul:

```c
merkle_level_kernel<<<level_blocks, threads>>>(...)
```

Pentru fiecare pozitie din nivelul urmator, un thread ia doua hash-uri din nivelul curent:

```c
left  = in_hashes + (2 * idx) * SHA256_HASH_SIZE;
right = in_hashes + (2 * idx + 1) * SHA256_HASH_SIZE;
```

Daca nu exista al doilea hash, atunci `right` devine egal cu `left`. In acest mod se respecta regula Merkle conform careia ultimul hash se dubleaza atunci cand numarul de noduri de pe un nivel este impar.

### 4.3. Concatenarea logica a hash-urilor

Pentru a calcula hash-ul unei perechi, nu construiesc un buffer separat cu cele doua hash-uri concatenate. In schimb, folosesc functia `apply_sha256_two_parts`, care actualizeaza acelasi context SHA-256 cu prima parte si apoi cu a doua parte:

```c
sha256_update(&ctx, a, len_a);
sha256_update(&ctx, b, len_b);
```

Rezultatul este acelasi ca pentru concatenarea efectiva, dar se evita o copiere intermediara.

### 4.4. Alternarea bufferelor

Pentru reducerea arborelui folosesc doua buffere, `g_d_hashes_a` si `g_d_hashes_b`. Un nivel citeste dintr-un buffer si scrie in celalalt. Dupa fiecare nivel, pointerii sunt schimbati:

```c
BYTE *tmp = in_hashes;
in_hashes = out_hashes;
out_hashes = tmp;
```

Aceasta tehnica simplifica implementarea si evita mutarea nivelurilor intermediare pe CPU.

La final, cand `current_n` devine 1, singurul hash ramas este copiat pe host in `merkle_root`.

---

## 5. Cautarea nonce-ului

### 5.1. Impartirea cautarii in batch-uri

Functia `find_nonce` cauta un nonce intre `0` si `max_nonce`. Pentru a nu lansa un kernel urias, cautarea este impartita in batch-uri:

```c
const uint64_t BATCH_SIZE = NONCE_BATCH_SIZE;
```

Pentru fiecare batch, se calculeaza numarul de blocuri CUDA necesare si se lanseaza `find_nonce_kernel`.

### 5.2. Contextul SHA-256 pentru prefix

Inainte de lansarea kernelului, partea fixa a blocului este introdusa intr-un context SHA-256:

```c
SHA256_CTX prefix_ctx;
sha256_init(&prefix_ctx);
sha256_update(&prefix_ctx, block_content, current_length);
```

Acest context este transmis kernelului. Fiecare thread primeste o copie a lui si adauga doar nonce-ul. Astfel, partea constanta a blocului nu este procesata de la zero pentru fiecare incercare.

### 5.3. Conversia nonce-ului

Nonce-ul trebuie concatenat la block content ca sir de caractere. Pe GPU folosesc functia:

```c
__device__ __forceinline__ int intToString(uint64_t num, char* out)
```

Aceasta converteste numarul in reprezentare zecimala. Am folosit aceasta functie in loc de `sprintf`, deoarece `sprintf` este prea costisitor pentru a fi apelat de foarte multe thread-uri CUDA.

### 5.4. Verificarea dificultatii

Kernelul nu compara direct stringuri hex complete pentru fiecare nonce. In schimb, functia:

```c
sha256_ctx_suffix_has_zero_prefix(...)
```

calculeaza digest-ul binar si verifica daca inceputul acestuia are numarul necesar de zerouri hex.

Pentru un numar par de zerouri, se verifica octeti intregi egali cu 0. Pentru un numar impar de zerouri, se verifica si nibble-ul superior al urmatorului octet.

Aceasta este o verificare mai directa si evita generarea completa a hash-ului in format text pentru fiecare nonce testat.

### 5.5. Selectarea rezultatului

Mai multe thread-uri pot gasi nonce-uri valide in acelasi batch. Pentru a pastra cel mai mic nonce valid, kernelul foloseste:

```c
atomicMin(best_nonce, nonce);
```

In plus, se seteaza flag-ul:

```c
atomicExch(found, 1);
```

Pe host, dupa fiecare kernel, se copiaza `found`. Daca este 1, inseamna ca batch-ul curent contine cel putin un nonce valid. Apoi se copiaza `best_nonce`, care reprezinta cel mai mic nonce valid din acel batch.

Batch-urile sunt parcurse crescator, deci primul batch in care `found` devine 1 este suficient pentru a obtine nonce-ul final.

Dupa gasirea nonce-ului, acesta este adaugat la `block_content`, iar hash-ul final al blocului este recalculat pe host.

---

## 6. Functii auxiliare

Am pastrat si adaptat mai multe functii auxiliare pentru a putea fi folosite atat pe host, cat si pe device:

- `d_strlen` - varianta simpla de `strlen`;
- `d_strcpy` - copiere de string pe device;
- `d_strcat` - concatenare de string pe device;
- `sha256_to_hex` - conversie digest binar in hash hex;
- `apply_sha256_len` - SHA-256 pentru input cu lungime cunoscuta;
- `apply_sha256` - SHA-256 pentru string terminat cu `\0`.

Aceste functii ajuta la pastrarea aceluiasi format al hash-urilor ca in implementarea CPU.

---

## 7. Warm-up GPU

Functia `warm_up_gpu` are doua roluri:

1. forteaza initializarea contextului CUDA prin lansarea unui kernel simplu;
2. prealoca bufferele principale folosite ulterior.

In plus, este lansat `clock_warmup_kernel`, care executa o bucla pe GPU. Scopul este ca GPU-ul sa fie deja activ inainte de masurarea functiilor importante.

Prealocarea foloseste dimensiuni fixe:

```c
#define PREALLOC_TRANSACTIONS_BYTES (64ULL * 1024ULL * 1024ULL)
#define PREALLOC_HASHES_BYTES (16ULL * 1024ULL * 1024ULL)
```

Astfel, pentru testele obisnuite, memoria este deja disponibila in momentul in care se proceseaza blocurile.

---

## 8. Corectitudine

Pentru corectitudine, implementarea respecta urmatoarele reguli:

- fiecare tranzactie este hash-uita fara caracterul nul final;
- hash-urile Merkle sunt combinate doua cate doua;
- la numar impar de hash-uri, ultimul element este duplicat;
- nivelurile Merkle sunt procesate pana ramane un singur hash;
- nonce-urile sunt generate in ordine crescatoare pe batch-uri;
- in interiorul unui batch, se retine cel mai mic nonce valid cu `atomicMin`;
- flag-ul `found` indica daca batch-ul curent contine cel putin un rezultat valid;
- hash-ul blocului este recalculat pe host dupa alegerea nonce-ului.

---

## 9. Compilare si rulare

Pentru compilare:

```bash
make
```

Pentru rulare:

```bash
make run TEST=test1
make run TEST=test2
make run TEST=test3
make run TEST=test4
```

Pentru curatare:

```bash
make clean
```

Testarea finala trebuie facuta pe infrastructura indicata in enunt, deoarece timpii depind de GPU-ul folosit.

---

## 10. Observatii despre performanta

Pentru Merkle root, paralelizarea este eficienta mai ales cand exista multe tranzactii in bloc. Hash-uirea initiala este complet independenta, iar nivelurile arborelui reduc treptat numarul de elemente.

Pentru nonce, performanta depinde de dificultate si de pozitia primului nonce valid. Daca nonce-ul valid apare devreme, sunt lansate putine batch-uri. Daca apare tarziu, se testeaza mai multe batch-uri.

Dimensiunea batch-ului reprezinta un compromis. Un batch mai mic permite verificarea mai frecventa a rezultatului pe host, dar creste numarul de lansari de kernel si numarul de transferuri mici. Un batch mai mare reduce overhead-ul de lansare, dar poate face mai multa munca dupa ce exista deja un nonce valid in batch.

In aceasta implementare am folosit `2^16` nonce-uri pe batch.

---

## 11. Limitari si imbunatatiri posibile

O prima imbunatatire posibila ar fi testarea mai multor valori pentru `NONCE_BATCH_SIZE`, deoarece dimensiunea optima depinde de test si de GPU.

O alta imbunatatire ar fi eliminarea flag-ului `found` si verificarea directa a valorii `best_nonce`, pentru a reduce un `cudaMemset` si un transfer host-device per batch. Totusi, varianta actuala este usor de inteles si separa clar ideea de "s-a gasit ceva" de valoarea nonce-ului minim.

Pentru Merkle root, o optimizare suplimentara ar fi tratarea nivelurilor foarte mici pe CPU sau combinarea unor pasi, pentru a reduce overhead-ul de kernel launch cand raman putine noduri.

---

## 12. Prompturi LLM folosite

Unealta folosita: **ChatGPT - GPT-5.5 Thinking**  
Scop: clarificarea unor concepte CUDA, intelegerea impactului unor decizii de implementare si redactarea documentatiei.

### Prompt 1

**Intrebare:**

> cum pot construi un Merkle root pe GPU daca am deja un vector de tranzactii si trebuie sa dublez ultimul hash cand numarul de elemente este impar?

**Raspuns primit, pe scurt:**

Modelul a explicat ca o abordare potrivita este sa se calculeze mai intai hash-ul fiecarei tranzactii in paralel, apoi sa se construiasca arborele nivel cu nivel. Pentru fiecare nivel, un thread poate procesa o pereche de hash-uri, iar daca perechea nu este completa, se foloseste de doua ori acelasi hash.

**Utilitate:**

Explicatia a fost utila pentru organizarea kernelului `merkle_level_kernel` si pentru validarea regulii de duplicare a ultimului hash.

---

### Prompt 2

**Intrebare:**

> dc se folosesc doua buffere pe device pentru construirea arborelui Merkle si cum se schimba pointerii intre niveluri?

**Raspuns primit, pe scurt:**

Modelul a descris tehnica ping-pong buffer: nivelul curent este citit dintr-un buffer, iar nivelul urmator este scris in celalalt. Dupa fiecare nivel, pointerii se interschimba. Astfel nu este nevoie sa se aloce memorie noua pentru fiecare pas si nu se copiaza nivelurile intermediare pe CPU.

**Utilitate:**

Raspunsul a ajutat la intelegerea motivului pentru care `g_d_hashes_a` si `g_d_hashes_b` sunt suficiente pentru intreaga constructie a arborelui.

---

### Prompt 3

**Intrebare:**

> cum pot cauta nonce-uri pe GPU in batch-uri si cum pot afla daca un batch contine un nonce valid?

**Raspuns primit, pe scurt:**

Modelul a explicat ca fiecare thread poate testa un nonce diferit, calculat ca `start_nonce + tid`. Daca un thread gaseste un nonce valid, poate seta un flag global cu `atomicExch`. Pentru a pastra cel mai mic nonce valid din batch, se poate folosi `atomicMin`.

**Utilitate:**

Raspunsul a fost util pentru structura kernelului `find_nonce_kernel`, unde folosesc atat `g_d_found`, cat si `g_d_best_nonce`.

---

### Prompt 4

**Intrebare:**

> dc este mai eficient sa initializez contextul SHA-256 pentru prefixul blocului o singura data si apoi sa adaug doar nonce-ul in kernel?

**Raspuns primit, pe scurt:**

Modelul a explicat ca partea fixa a blocului este aceeasi pentru toate nonce-urile. Daca aceasta este introdusa o singura data intr-un context SHA-256, fiecare thread trebuie sa proceseze doar partea variabila, adica nonce-ul. Acest lucru reduce munca repetata pentru fiecare incercare.

**Utilitate:**

Explicatia a clarificat optimizarea folosita in `find_nonce`, unde `prefix_ctx` este pregatit pe host si transmis catre kernel.

---

### Prompt 5

**Intrebare:**

> cce ar trebui sa explic in README pentru o tema CUDA cu Merkle root si Proof of Work ca sa fie clar la evaluare?

**Raspuns primit, pe scurt:**

Modelul a recomandat sa fie explicate separat fluxul pentru Merkle root, fluxul pentru nonce, alocarile pe GPU, folosirea operatiilor atomice, verificarile de corectitudine si modul de testare. De asemenea, a recomandat mentionarea deciziilor manuale si a limitarilor implementarii.

**Utilitate:**

Raspunsul a fost util pentru structurarea acestui README si pentru formularea explicatiilor intr-un mod mai usor de urmarit.

---

## 13. Decizii luate manual

In implementarea finala am ales manual urmatoarele aspecte:

- folosirea a 256 de thread-uri per block pentru kernelurile principale;
- pastrarea bufferelor globale pentru a evita alocarile repetate;
- impartirea cautarii nonce-ului in batch-uri;
- folosirea flag-ului `found` pentru a sti rapid daca un batch contine rezultat;
- folosirea lui `atomicMin` pentru a alege cel mai mic nonce valid;
- conversia nonce-ului pe device cu o functie proprie, nu cu `sprintf`;
- recalcularea hash-ului final al blocului pe host dupa gasirea nonce-ului;
- folosirea unei functii de warm-up pentru initializarea si pregatirea GPU-ului.

---

## 14. Concluzie

Solutia muta pe GPU operatiile care se repeta de foarte multe ori: hash-uirea tranzactiilor, reducerea Merkle si testarea nonce-urilor. Prin folosirea bufferelor globale, a reducerii pe niveluri si a cautarii in batch-uri, implementarea reduce timpul petrecut in calcule seriale pe CPU.

Varianta este gandita pentru a fi clara, usor de verificat si compatibila cu cerintele temei. Performanta exacta trebuie evaluata prin rularea testelor pe infrastructura ceruta.
