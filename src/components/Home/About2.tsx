
import vector3 from "../../assests/vector3.jpg";
const bgstyle = {
  backgroundImage: `url(${vector3})`,
  height: "100%",
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const About2 = () => {
  return (
   <>
    <div className="relative py-14">
        <div className="container py-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 place-items-center gap-4">
            <div className='order-2 sm:order-1'>
              <div className="text-3xl font-semibold text-cyan-950 ">
                AI Recognition
              </div>
              <p className="py-4 font-semibold text-cyan-950">
                Powered by advanced machine learning algorithms, our AI
                recognition engine can distinguish between various types of
                waste, from common plastics to complex electronics. This
                precision helps you sort correctly, reducing contamination and
                maximizing recycling efforts.
              </p>
            </div>

            <div
              style={bgstyle}
              className="rounded-3xl overflow-hidden  shadow-2xl order-1 sm:order-2"
            >
              <img src={vector3} alt="" className="w-full " />
            </div>
          </div>
        </div>
      </div>
   </>
  )
}

export default About2
