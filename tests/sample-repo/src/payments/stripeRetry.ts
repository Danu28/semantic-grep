export async function stripeRetry(charge: any) {
  // exponential backoff for charge failures
  for(let i=0;i<5;i++){ try{ return await charge(); } catch(e){ await new Promise(r=>setTimeout(r, 2**i*100)); } }
}
