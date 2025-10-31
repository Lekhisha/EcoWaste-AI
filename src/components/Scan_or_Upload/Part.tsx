
import image6 from "../../assests/image6.jpg";

const bgstyle2 = {
  backgroundImage: `url(${image6})`,
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const Part = () => {
  return (
    <div className='Relative z-[-1] py-20 bg-linear-to-r from-amber-200 to-green-500'>
      <div className='container py-4'>
        <div className='grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center'>
            <div className='order-2 sm:order-1'>
                <div className='text-5xl text-cyan-950 font-bold'><span className='text-green-600'>Revolutionize</span> Waste Classification With AI</div>
                <p className='text-cyan-950 font-semibold py-4'>Identify, sort, and recycle any item effortlessly. Our AI-powered scanner provides instant insights, helping you make eco-friendly choices every day.</p>
                <button className="font-bold bg-green-700 text-white rounded py-2 px-5">
                Get Started Now
              </button>
            </div>
            <div className='rounded-3xl overflow-hidden shadow-2xl h-72 sm:h-96 md:h-[350px] order-1 sm:order-2' style={bgstyle2}>
               
            </div>
        </div>
      </div>
    </div>
  )
}

export default Part
